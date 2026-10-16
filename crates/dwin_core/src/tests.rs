//! Tests covering all 12 acceptance card items plus mock tests for each trait.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use async_trait::async_trait;

use crate::traits::futures_like::Stream;
use crate::traits::EventStreamExt;
use crate::{
    AgentError, AgentLoop, Approval, Approver, AuditRecord, Decision, EventStream, Hook,
    HookContext, HookEvent, HookOutcome, NullOutputHandler, Provider, ProviderEvent, Registry,
    Request, RoleDna, SessionEntry, Store, Tool, ToolCall, ToolOutput, ToolSpec, DateHook,
};

// =============================================================================
// Mock Implementations
// =============================================================================

/// Mock provider that returns a sequence of events.
struct MockProvider {
    responses: Arc<Mutex<Vec<Vec<ProviderEvent>>>>,
    call_count: Arc<AtomicUsize>,
}

impl MockProvider {
    fn new(responses: Vec<Vec<ProviderEvent>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

struct MockEventStream {
    events: Vec<ProviderEvent>,
    index: usize,
}

impl Stream for MockEventStream {
    type Item = ProviderEvent;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.index < self.events.len() {
            let event = self.events[self.index].clone();
            self.index += 1;
            Poll::Ready(Some(event))
        } else {
            Poll::Ready(None)
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn stream(&self, _req: Request) -> anyhow::Result<EventStream> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut responses = self.responses.lock().unwrap();
        let events = if responses.is_empty() {
            vec![ProviderEvent::Done {
                usage: Default::default(),
            }]
        } else {
            responses.remove(0)
        };
        Ok(Box::pin(MockEventStream { events, index: 0 }))
    }
}

/// Mock provider that always requests a tool call (for testing 25-cap).
#[derive(Clone)]
struct AlwaysToolCallingProvider {
    call_count: Arc<AtomicUsize>,
}

impl AlwaysToolCallingProvider {
    fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for AlwaysToolCallingProvider {
    async fn stream(&self, _req: Request) -> anyhow::Result<EventStream> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        let events = vec![
            ProviderEvent::ToolCall(ToolCall {
                id: format!("call_{}", count),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ];
        Ok(Box::pin(MockEventStream { events, index: 0 }))
    }
}

/// Mock tool that records calls.
struct MockTool {
    name: String,
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
    response: String,
}

impl MockTool {
    fn new(name: &str, response: &str) -> Self {
        Self {
            name: name.to_string(),
            calls: Arc::new(Mutex::new(Vec::new())),
            response: response.to_string(),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl Tool for MockTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: format!("Mock tool: {}", self.name),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                }
            }),
        }
    }

    async fn call(&self, args: serde_json::Value) -> anyhow::Result<ToolOutput> {
        self.calls.lock().unwrap().push(args);
        Ok(ToolOutput::success(&self.response))
    }
}

/// Mock store that records audit entries.
#[derive(Clone)]
struct MockStore {
    entries: Arc<Mutex<HashMap<String, Vec<SessionEntry>>>>,
    audits: Arc<Mutex<Vec<AuditRecord>>>,
}

impl MockStore {
    fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            audits: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn audit_count(&self) -> usize {
        self.audits.lock().unwrap().len()
    }

    fn audits(&self) -> Vec<AuditRecord> {
        self.audits.lock().unwrap().clone()
    }
}

#[async_trait]
impl Store for MockStore {
    async fn append(&self, session_id: &str, entry: SessionEntry) -> anyhow::Result<()> {
        self.entries
            .lock()
            .unwrap()
            .entry(session_id.to_string())
            .or_default()
            .push(entry);
        Ok(())
    }

    async fn load(&self, session_id: &str) -> anyhow::Result<Vec<SessionEntry>> {
        Ok(self
            .entries
            .lock()
            .unwrap()
            .get(session_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn list(&self) -> anyhow::Result<Vec<String>> {
        Ok(self.entries.lock().unwrap().keys().cloned().collect())
    }

    async fn audit(&self, record: AuditRecord) -> anyhow::Result<()> {
        self.audits.lock().unwrap().push(record);
        Ok(())
    }
}

/// Mock hook that returns a specific decision for BeforeToolCall.
struct MockHook {
    decision: Decision,
    called: Arc<AtomicUsize>,
}

impl MockHook {
    fn new(decision: Decision) -> Self {
        Self {
            decision,
            called: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn call_count(&self) -> usize {
        self.called.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Hook for MockHook {
    async fn on<'a>(
        &self,
        event: &HookEvent<'a>,
        _ctx: &mut HookContext<'a>,
    ) -> anyhow::Result<HookOutcome> {
        if matches!(event, HookEvent::BeforeToolCall { .. }) {
            self.called.fetch_add(1, Ordering::SeqCst);
            Ok(HookOutcome::Decision(self.decision.clone()))
        } else {
            Ok(HookOutcome::Continue)
        }
    }
}

/// Mock approver that returns a specific approval.
#[derive(Clone)]
struct MockApprover {
    approve: bool,
    approver_name: String,
    asked: Arc<AtomicUsize>,
}

impl MockApprover {
    fn new(approve: bool, approver_name: &str) -> Self {
        Self {
            approve,
            approver_name: approver_name.to_string(),
            asked: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn ask_count(&self) -> usize {
        self.asked.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Approver for MockApprover {
    async fn ask(&self, _call: &ToolCall, _reason: &str) -> anyhow::Result<Approval> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        if self.approve {
            Ok(Approval::approved(&self.approver_name))
        } else {
            Ok(Approval::denied(&self.approver_name))
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_role_dna() -> RoleDna {
    RoleDna::new("test@fene.ae", "test", "Test Role", "You are a test assistant.")
        .with_tools(["read", "write"])
}

fn create_read_tool() -> MockTool {
    MockTool::new("read", "file contents")
}

// =============================================================================
// Card Test 1: Two-iteration turn
// =============================================================================

#[tokio::test]
async fn test_card1_two_iteration_turn() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        // First iteration: request a tool call
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        // Second iteration: respond with text
        vec![
            ProviderEvent::TextDelta("The file contains: file contents".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(result.iterations, 2, "Should complete in exactly 2 iterations");
    assert_eq!(tool.call_count(), 1, "Tool should be called exactly once");
    assert!(result.completed, "Turn should complete normally");
}

// =============================================================================
// Card Test 2: Deny blocks tool
// =============================================================================

#[tokio::test]
async fn test_card2_deny_blocks_tool() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        // First iteration: request a tool call
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        // Second iteration: model acknowledges the denial
        vec![
            ProviderEvent::TextDelta("I cannot read that file: blocked".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());
    registry.register_hook(MockHook::new(Decision::Deny("blocked".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(tool.call_count(), 0, "Tool should NOT be called when denied");
    // Check that the tool result message contains "blocked"
    let messages = agent.messages();
    let tool_message = messages.iter().find(|m| m.tool_call_id.is_some()).unwrap();
    assert!(
        tool_message.content.contains("blocked"),
        "Denial reason should be in tool result: {}",
        tool_message.content
    );
    assert!(result.completed);
}

// =============================================================================
// Card Test 3: RequireApproval yes - tool runs
// =============================================================================

#[tokio::test]
async fn test_card3_require_approval_yes() {
    let tool = Arc::new(create_read_tool());
    let approver = MockApprover::new(true, "human:alice");
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Done".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());
    registry.register_hook(MockHook::new(Decision::RequireApproval(
        "needs approval".to_string(),
    )));

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver.clone(),
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(approver.ask_count(), 1, "Approver should be asked exactly once");
    assert_eq!(tool.call_count(), 1, "Tool should run after approval");
}

// =============================================================================
// Card Test 4: RequireApproval no - tool not run
// =============================================================================

#[tokio::test]
async fn test_card4_require_approval_no() {
    let tool = Arc::new(create_read_tool());
    let approver = MockApprover::new(false, "human:bob");
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Understood, approval was denied".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());
    registry.register_hook(MockHook::new(Decision::RequireApproval(
        "needs approval".to_string(),
    )));

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(tool.call_count(), 0, "Tool should NOT run when approval denied");
}

// =============================================================================
// Card Test 5: Most restrictive wins
// =============================================================================

#[tokio::test]
async fn test_card5_most_restrictive_wins_deny() {
    // Allow + RequireApproval + Deny => Deny
    let tool = Arc::new(create_read_tool());
    let approver = MockApprover::new(true, "human:test");
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Denied".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());
    registry.register_hook(MockHook::new(Decision::Allow));
    registry.register_hook(MockHook::new(Decision::RequireApproval("maybe".to_string())));
    registry.register_hook(MockHook::new(Decision::Deny("no way".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver.clone(),
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(tool.call_count(), 0, "Tool should NOT run when Deny wins");
    assert_eq!(approver.ask_count(), 0, "Approver should NOT be asked when Deny wins");
}

#[tokio::test]
async fn test_card5_most_restrictive_wins_require_approval() {
    // Allow + RequireApproval => RequireApproval (Approver asked)
    let tool = Arc::new(create_read_tool());
    let approver = MockApprover::new(true, "human:test");
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Approved".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let mut registry = Registry::new();
    registry.register_tool_arc(tool.clone());
    registry.register_hook(MockHook::new(Decision::Allow));
    registry.register_hook(MockHook::new(Decision::RequireApproval("check first".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver.clone(),
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _result = agent.run_turn("Read test.txt").await.unwrap();

    assert_eq!(approver.ask_count(), 1, "Approver should be asked when RequireApproval wins");
    assert_eq!(tool.call_count(), 1, "Tool should run after approval");
}

// =============================================================================
// Card Test 6: Allowlist - non-allowlisted tool denied
// =============================================================================

#[tokio::test]
async fn test_card6_allowlist_denies_non_allowlisted() {
    let read_tool = Arc::new(MockTool::new("read", "content"));
    let bash_tool = Arc::new(MockTool::new("bash", "output"));
    
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "bash".to_string(),
                args: serde_json::json!({"command": "ls"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Cannot run bash".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(read_tool);
    registry.register_tool_arc(bash_tool.clone());
    // All hooks allow
    registry.register_hook(MockHook::new(Decision::Allow));

    // Role DNA only allows "read", not "bash"
    let role_dna = RoleDna::new("test@fene.ae", "test", "Test", "Test")
        .with_tool("read");

    let mut agent = AgentLoop::new(
        provider,
        store.clone(),
        approver,
        NullOutputHandler,
        registry,
        role_dna,
        "session_1",
    );

    let _result = agent.run_turn("Run ls").await.unwrap();

    assert_eq!(bash_tool.call_count(), 0, "Non-allowlisted tool should NOT run");
    
    // Check audit shows denial
    let audits = store.audits();
    assert_eq!(audits.len(), 1);
    assert_eq!(audits[0].decision, "denied");
    assert!(audits[0].reason.as_ref().unwrap().contains("not in the allowlist"));
}

#[tokio::test]
async fn test_card6_allowlist_specs_not_sent_for_non_allowlisted() {
    // Verify that tool specs for non-allowlisted tools are not sent to provider
    let registry = {
        let mut r = Registry::new();
        r.register_tool(MockTool::new("read", ""));
        r.register_tool(MockTool::new("write", ""));
        r.register_tool(MockTool::new("bash", ""));
        r
    };

    let role_dna = RoleDna::new("test@fene.ae", "test", "Test", "Test")
        .with_tools(["read", "write"]); // bash NOT in allowlist

    let specs = registry.get_tool_specs(&role_dna.tools);
    let spec_names: Vec<_> = specs.iter().map(|s| s.name.as_str()).collect();

    assert!(spec_names.contains(&"read"));
    assert!(spec_names.contains(&"write"));
    assert!(!spec_names.contains(&"bash"), "bash should NOT be in tool specs");
}

// =============================================================================
// Card Test 7: 25-iteration cap
// =============================================================================

#[tokio::test]
async fn test_card7_25_iteration_cap() {
    let tool = Arc::new(create_read_tool());
    let provider = AlwaysToolCallingProvider::new();
    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);

    let mut agent = AgentLoop::new(
        provider,
        store,
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let result = agent.run_turn("Loop forever").await;

    match result {
        Err(AgentError::MaxIterationsExceeded) => {
            // Expected - this is the clean error
        }
        Ok(_) => panic!("Should have hit max iterations"),
        Err(e) => panic!("Wrong error type: {:?}", e),
    }
}

#[tokio::test]
async fn test_card7_25_cap_exact_count() {
    let tool = Arc::new(create_read_tool());
    let provider = AlwaysToolCallingProvider::new();
    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);

    let mut agent = AgentLoop::new(
        provider.clone(),
        store,
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _ = agent.run_turn("Loop").await;

    // Provider should be called exactly 25 times (the cap)
    // The error occurs when we would exceed 25, so we make 25 calls then error
    assert_eq!(
        provider.call_count(),
        25,
        "Provider should be called exactly 25 times before cap"
    );
}

// =============================================================================
// Card Test 8: Neutral Request - system prompt in own field
// =============================================================================

#[tokio::test]
async fn test_card8_neutral_request_system_prompt_separate() {
    use crate::Request;

    let request = Request::new("You are an assistant.")
        .with_messages(vec![crate::Message::user("Hello")])
        .with_tools(vec![]);

    // System prompt is in its own field
    assert_eq!(request.system_prompt, "You are an assistant.");
    // Messages don't contain the system prompt
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].content, "Hello");
    // No system message in the messages array
    assert!(!request
        .messages
        .iter()
        .any(|m| m.role == crate::MessageRole::System));
}

// =============================================================================
// Card Test 9: Audit - every tool outcome produces exactly one audit record
// =============================================================================

#[tokio::test]
async fn test_card9_audit_allowed_tool() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Done".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);

    let mut agent = AgentLoop::new(
        provider,
        store.clone(),
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _ = agent.run_turn("Read file").await.unwrap();

    let audits = store.audits();
    assert_eq!(audits.len(), 1, "Exactly one audit record for one tool call");
    assert_eq!(audits[0].agent_id, "test@fene.ae");
    assert_eq!(audits[0].role, "test");
    assert_eq!(audits[0].tool, "read");
    assert_eq!(audits[0].decision, "allowed");
    assert_eq!(audits[0].outcome, "success");
}

#[tokio::test]
async fn test_card9_audit_denied_tool() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "secret.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Denied".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "test");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);
    registry.register_hook(MockHook::new(Decision::Deny("forbidden".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store.clone(),
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _ = agent.run_turn("Read secret").await.unwrap();

    let audits = store.audits();
    assert_eq!(audits.len(), 1, "Exactly one audit record for denied tool");
    assert_eq!(audits[0].decision, "denied");
    assert_eq!(audits[0].reason.as_ref().unwrap(), "forbidden");
    assert_eq!(audits[0].approver.as_ref().unwrap(), "policy");
    assert_eq!(audits[0].outcome, "denied");
}

#[tokio::test]
async fn test_card9_audit_approved_tool() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Done".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(true, "human:alice");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);
    registry.register_hook(MockHook::new(Decision::RequireApproval("risky".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store.clone(),
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _ = agent.run_turn("Read file").await.unwrap();

    let audits = store.audits();
    assert_eq!(audits.len(), 1);
    assert_eq!(audits[0].decision, "approved");
    assert_eq!(audits[0].approver.as_ref().unwrap(), "human:alice");
    assert_eq!(audits[0].outcome, "success");
}

#[tokio::test]
async fn test_card9_audit_rejected_tool() {
    let tool = Arc::new(create_read_tool());
    let provider = MockProvider::new(vec![
        vec![
            ProviderEvent::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                args: serde_json::json!({"path": "test.txt"}),
            }),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
        vec![
            ProviderEvent::TextDelta("Rejected".to_string()),
            ProviderEvent::Done {
                usage: Default::default(),
            },
        ],
    ]);

    let store = MockStore::new();
    let approver = MockApprover::new(false, "human:bob");
    let mut registry = Registry::new();
    registry.register_tool_arc(tool);
    registry.register_hook(MockHook::new(Decision::RequireApproval("risky".to_string())));

    let mut agent = AgentLoop::new(
        provider,
        store.clone(),
        approver,
        NullOutputHandler,
        registry,
        create_test_role_dna(),
        "session_1",
    );

    let _ = agent.run_turn("Read file").await.unwrap();

    let audits = store.audits();
    assert_eq!(audits.len(), 1);
    assert_eq!(audits[0].decision, "rejected");
    assert_eq!(audits[0].approver.as_ref().unwrap(), "human:bob");
    assert_eq!(audits[0].outcome, "denied");
}

// =============================================================================
// Card Test 10: Date hook appends today's date
// =============================================================================

#[tokio::test]
async fn test_card10_date_hook_appends_date() {
    use chrono::Utc;

    let hook = DateHook::new();
    let mut system_prompt = "You are an assistant.".to_string();
    let mut ctx = HookContext {
        system_prompt: &mut system_prompt,
        agent_id: "test",
        role: "test",
    };
    let request = Request::new("");
    let event = HookEvent::BeforeRequest { request: &request };

    hook.on(&event, &mut ctx).await.unwrap();

    let today = Utc::now().format("%Y-%m-%d").to_string();
    assert!(
        ctx.system_prompt.contains(&format!("Current date: {}", today)),
        "System prompt should contain today's date: {}",
        ctx.system_prompt
    );
}

// =============================================================================
// Card Test 11: Isolation - no workspace crates, no reqwest/rusqlite/std::fs
// =============================================================================

// This is verified by `cargo tree -p core` having no workspace crates
// and no reqwest/rusqlite. The test here documents the requirement.
#[test]
fn test_card11_isolation_documented() {
    // This test documents that core has no dependencies on:
    // - Other workspace crates (providers, tools, store, policy, cli)
    // - reqwest
    // - rusqlite
    // - std::fs (no file I/O)
    //
    // Verification: run `cargo tree -p core` and check the output
    // The CI will run this check.
}

// =============================================================================
// Card Test 12: No warnings
// =============================================================================

// This is verified by `cargo clippy -p core -- -D warnings`
// The test here documents the requirement.
#[test]
fn test_card12_no_warnings_documented() {
    // This test documents that:
    // - cargo build produces no warnings
    // - cargo clippy -p core -- -D warnings passes
    //
    // Verification: run these commands in CI
}

// =============================================================================
// Trait Mock Tests
// =============================================================================

#[tokio::test]
async fn test_mock_provider_trait() {
    let provider = MockProvider::new(vec![vec![
        ProviderEvent::TextDelta("Hello".to_string()),
        ProviderEvent::Done {
            usage: Default::default(),
        },
    ]]);

    let request = Request::new("Test");
    let mut stream = provider.stream(request).await.unwrap();

    let event = stream.next().await.unwrap();
    assert!(matches!(event, ProviderEvent::TextDelta(t) if t == "Hello"));
}

#[tokio::test]
async fn test_mock_tool_trait() {
    let tool = MockTool::new("test", "result");
    let spec = tool.spec();
    assert_eq!(spec.name, "test");

    let output = tool.call(serde_json::json!({"key": "value"})).await.unwrap();
    assert_eq!(output.content, "result");
    assert!(!output.is_error);
}

#[tokio::test]
async fn test_mock_store_trait() {
    let store = MockStore::new();

    let entry = SessionEntry {
        agent_id: "test".to_string(),
        role: "test".to_string(),
        timestamp: 0,
        msg_role: crate::MessageRole::User,
        payload: serde_json::json!({"content": "hello"}),
    };

    store.append("session_1", entry).await.unwrap();
    let entries = store.load("session_1").await.unwrap();
    assert_eq!(entries.len(), 1);

    let sessions = store.list().await.unwrap();
    assert!(sessions.contains(&"session_1".to_string()));

    let audit = AuditRecord::new("agent", "role", "tool", serde_json::json!({}));
    store.audit(audit).await.unwrap();
    assert_eq!(store.audit_count(), 1);
}

#[tokio::test]
async fn test_mock_hook_trait() {
    let hook = MockHook::new(Decision::Allow);
    let mut system_prompt = "test".to_string();
    let mut ctx = HookContext {
        system_prompt: &mut system_prompt,
        agent_id: "test",
        role: "test",
    };

    let call = ToolCall {
        id: "1".to_string(),
        name: "test".to_string(),
        args: serde_json::json!({}),
    };
    let event = HookEvent::BeforeToolCall {
        call: &call,
        tier: None,
    };

    let outcome = hook.on(&event, &mut ctx).await.unwrap();
    assert!(matches!(outcome, HookOutcome::Decision(Decision::Allow)));
}

#[tokio::test]
async fn test_mock_approver_trait() {
    let approver = MockApprover::new(true, "tester");
    let call = ToolCall {
        id: "1".to_string(),
        name: "test".to_string(),
        args: serde_json::json!({}),
    };

    let approval = approver.ask(&call, "reason").await.unwrap();
    assert!(approval.approved);
    assert_eq!(approval.approver, "tester");
}
