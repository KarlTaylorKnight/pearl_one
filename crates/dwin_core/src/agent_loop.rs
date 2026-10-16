//! Agent loop implementation with allowlist filtering and 25-iteration cap.

use crate::{
    Approver, AuditRecord, Decision, EventStreamExt, HookContext, HookEvent,
    Message, Provider, ProviderEvent, Registry, Request, RoleDna, Store, ToolCall,
    ToolCallInfo, ToolOutput,
};

/// Maximum iterations per turn to prevent infinite loops.
pub const MAX_ITERATIONS: usize = 25;

/// Error type for agent loop operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Maximum iterations ({MAX_ITERATIONS}) exceeded")]
    MaxIterationsExceeded,

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Hook error: {0}")]
    HookError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result of a single turn in the agent loop.
#[derive(Debug)]
pub struct TurnResult {
    /// The assistant's text response (if any).
    pub text: String,
    /// Number of iterations used.
    pub iterations: usize,
    /// Whether the turn completed normally (no tool calls pending).
    pub completed: bool,
}

/// Output handler for streaming text and events.
#[async_trait::async_trait]
pub trait OutputHandler: Send + Sync {
    /// Called when text is received from the provider.
    async fn on_text(&self, text: &str);
    /// Called when a tool call is about to be executed.
    async fn on_tool_call(&self, call: &ToolCall);
    /// Called when a tool call completes.
    async fn on_tool_result(&self, call: &ToolCall, output: &ToolOutput);
}

/// Default output handler that does nothing (for testing).
pub struct NullOutputHandler;

#[async_trait::async_trait]
impl OutputHandler for NullOutputHandler {
    async fn on_text(&self, _text: &str) {}
    async fn on_tool_call(&self, _call: &ToolCall) {}
    async fn on_tool_result(&self, _call: &ToolCall, _output: &ToolOutput) {}
}

/// The agent loop runner.
pub struct AgentLoop<P, S, A, O> {
    provider: P,
    store: S,
    approver: A,
    output: O,
    registry: Registry,
    role_dna: RoleDna,
    session_id: String,
    messages: Vec<Message>,
}

impl<P, S, A, O> AgentLoop<P, S, A, O> {
    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the role DNA.
    pub fn role_dna(&self) -> &RoleDna {
        &self.role_dna
    }
}

impl<P, S, A, O> AgentLoop<P, S, A, O>
where
    P: Provider,
    S: Store,
    A: Approver,
    O: OutputHandler,
{
    /// Create a new agent loop.
    pub fn new(
        provider: P,
        store: S,
        approver: A,
        output: O,
        registry: Registry,
        role_dna: RoleDna,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            store,
            approver,
            output,
            registry,
            role_dna,
            session_id: session_id.into(),
            messages: Vec::new(),
        }
    }

    /// Get the current messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Set the messages (e.g., when resuming a session).
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Add a user message and run the agent loop until completion or max iterations.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<TurnResult, AgentError> {
        self.messages.push(Message::user(user_input));

        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                return Err(AgentError::MaxIterationsExceeded);
            }

            let mut system_prompt = self.role_dna.system_prompt.clone();

            // Run BeforeRequest hooks to modify system prompt
            for hook in self.registry.hooks() {
                let request = Request::new(&system_prompt);
                let mut ctx = HookContext {
                    system_prompt: &mut system_prompt,
                    agent_id: &self.role_dna.agent_id,
                    role: &self.role_dna.role,
                };
                let event = HookEvent::BeforeRequest { request: &request };
                hook.on(&event, &mut ctx)
                    .await
                    .map_err(|e| AgentError::HookError(e.to_string()))?;
            }

            let tool_specs = self.registry.get_tool_specs(&self.role_dna.tools);

            let request = Request::new(&system_prompt)
                .with_messages(self.messages.clone())
                .with_tools(tool_specs);

            let mut stream = self
                .provider
                .stream(request)
                .await
                .map_err(|e| AgentError::ProviderError(e.to_string()))?;

            let mut text_buffer = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();

            while let Some(event) = stream.next().await {
                match event {
                    ProviderEvent::TextDelta(text) => {
                        self.output.on_text(&text).await;
                        text_buffer.push_str(&text);
                    }
                    ProviderEvent::ToolCall(call) => {
                        tool_calls.push(call);
                    }
                    ProviderEvent::Done { .. } => {
                        break;
                    }
                    ProviderEvent::Error(e) => {
                        return Err(AgentError::ProviderError(e));
                    }
                }
            }

            if tool_calls.is_empty() {
                if !text_buffer.is_empty() {
                    self.messages.push(Message::assistant(&text_buffer));
                }
                return Ok(TurnResult {
                    text: text_buffer,
                    iterations,
                    completed: true,
                });
            }

            let tool_call_infos: Vec<ToolCallInfo> = tool_calls
                .iter()
                .map(|tc| ToolCallInfo {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.args.to_string(),
                })
                .collect();
            self.messages
                .push(Message::assistant_with_tool_calls(&text_buffer, tool_call_infos));

            for call in tool_calls {
                let tool_result = self.process_tool_call(&call).await?;
                self.messages
                    .push(Message::tool(&call.id, &tool_result.content));
            }
        }
    }

    /// Process a single tool call, handling allowlist, hooks, approval, and auditing.
    async fn process_tool_call(&mut self, call: &ToolCall) -> Result<ToolOutput, AgentError> {
        let mut audit = AuditRecord::new(
            &self.role_dna.agent_id,
            &self.role_dna.role,
            &call.name,
            call.args.clone(),
        );

        // Get tier from role DNA or registry (needed for all audit records)
        let tier = self
            .role_dna
            .get_risk_tier(&call.name)
            .or_else(|| self.registry.get_risk_tier(&call.name));
        if let Some(t) = tier {
            audit = audit.with_tier(t);
        }

        if !self.role_dna.is_tool_allowed(&call.name) {
            let reason = format!("Tool '{}' is not in the allowlist", call.name);
            audit = audit
                .with_decision("denied")
                .with_reason(&reason)
                .with_approver("policy")
                .with_outcome("denied");
            self.store
                .audit(audit)
                .await
                .map_err(|e| AgentError::StoreError(e.to_string()))?;
            return Ok(ToolOutput::error(&reason));
        }

        let mut decisions = Vec::new();
        for hook in self.registry.hooks() {
            let mut system_prompt = self.role_dna.system_prompt.clone();
            let mut ctx = HookContext {
                system_prompt: &mut system_prompt,
                agent_id: &self.role_dna.agent_id,
                role: &self.role_dna.role,
            };
            let event = HookEvent::BeforeToolCall { call, tier };
            let outcome = hook
                .on(&event, &mut ctx)
                .await
                .map_err(|e| AgentError::HookError(e.to_string()))?;
            if let Some(decision) = outcome.into_decision() {
                decisions.push(decision);
            }
        }

        let final_decision = Decision::combine(decisions);

        match &final_decision {
            Decision::Allow => {
                let reason = match tier {
                    Some(t) => format!("policy: tier {} auto-allow", t),
                    None => "policy: auto-allow".to_string(),
                };
                audit = audit
                    .with_decision("allowed")
                    .with_reason(&reason)
                    .with_approver("policy");
                let output = self.execute_tool_safe(call).await;
                audit = audit.with_outcome(if output.is_error { "error" } else { "success" });
                self.store
                    .audit(audit)
                    .await
                    .map_err(|e| AgentError::StoreError(e.to_string()))?;
                self.emit_after_tool_call(call, &output).await?;
                Ok(output)
            }
            Decision::Deny(reason) => {
                audit = audit
                    .with_decision("denied")
                    .with_reason(reason)
                    .with_approver("policy")
                    .with_outcome("denied");
                self.store
                    .audit(audit)
                    .await
                    .map_err(|e| AgentError::StoreError(e.to_string()))?;
                Ok(ToolOutput::error(reason))
            }
            Decision::RequireApproval(reason) => {
                self.output.on_tool_call(call).await;
                let approval = self
                    .approver
                    .ask(call, reason)
                    .await
                    .map_err(|e| AgentError::Internal(e.to_string()))?;

                if approval.approved {
                    audit = audit
                        .with_decision("approved")
                        .with_reason(reason)
                        .with_approver(&approval.approver);
                    let output = self.execute_tool_safe(call).await;
                    audit = audit.with_outcome(if output.is_error { "error" } else { "success" });
                    self.store
                        .audit(audit)
                        .await
                        .map_err(|e| AgentError::StoreError(e.to_string()))?;
                    self.emit_after_tool_call(call, &output).await?;
                    Ok(output)
                } else {
                    let deny_reason = format!("Approval denied by {}: {}", approval.approver, reason);
                    audit = audit
                        .with_decision("rejected")
                        .with_reason(reason)
                        .with_approver(&approval.approver)
                        .with_outcome("denied");
                    self.store
                        .audit(audit)
                        .await
                        .map_err(|e| AgentError::StoreError(e.to_string()))?;
                    Ok(ToolOutput::error(&deny_reason))
                }
            }
        }
    }

    /// Execute a tool and return its output.
    /// Returns a ToolOutput even if the tool returns an error (with is_error=true).
    async fn execute_tool_safe(&self, call: &ToolCall) -> ToolOutput {
        let tool = match self.registry.get_tool(&call.name) {
            Some(t) => t,
            None => return ToolOutput::error(format!("Tool '{}' not found", call.name)),
        };

        self.output.on_tool_call(call).await;
        let output = match tool.call(call.args.clone()).await {
            Ok(out) => out,
            Err(e) => ToolOutput::error(format!("Tool error: {}", e)),
        };
        self.output.on_tool_result(call, &output).await;
        output
    }

    /// Emit AfterToolCall hook event.
    async fn emit_after_tool_call(
        &self,
        call: &ToolCall,
        output: &ToolOutput,
    ) -> Result<(), AgentError> {
        for hook in self.registry.hooks() {
            let mut system_prompt = self.role_dna.system_prompt.clone();
            let mut ctx = HookContext {
                system_prompt: &mut system_prompt,
                agent_id: &self.role_dna.agent_id,
                role: &self.role_dna.role,
            };
            let event = HookEvent::AfterToolCall { call, output };
            hook.on(&event, &mut ctx)
                .await
                .map_err(|e| AgentError::HookError(e.to_string()))?;
        }
        Ok(())
    }
}

/// Builder for AgentLoop with optional components.
pub struct AgentLoopBuilder<P, S, A, O> {
    provider: Option<P>,
    store: Option<S>,
    approver: Option<A>,
    output: Option<O>,
    registry: Registry,
    role_dna: Option<RoleDna>,
    session_id: String,
}

impl<P, S, A, O> AgentLoopBuilder<P, S, A, O> {
    pub fn new() -> Self {
        Self {
            provider: None,
            store: None,
            approver: None,
            output: None,
            registry: Registry::new(),
            role_dna: None,
            session_id: String::new(),
        }
    }

    pub fn provider(mut self, provider: P) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn store(mut self, store: S) -> Self {
        self.store = Some(store);
        self
    }

    pub fn approver(mut self, approver: A) -> Self {
        self.approver = Some(approver);
        self
    }

    pub fn output(mut self, output: O) -> Self {
        self.output = Some(output);
        self
    }

    pub fn registry(mut self, registry: Registry) -> Self {
        self.registry = registry;
        self
    }

    pub fn role_dna(mut self, role_dna: RoleDna) -> Self {
        self.role_dna = Some(role_dna);
        self
    }

    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }
}

impl<P, S, A, O> AgentLoopBuilder<P, S, A, O>
where
    P: Provider,
    S: Store,
    A: Approver,
    O: OutputHandler,
{
    pub fn build(self) -> Result<AgentLoop<P, S, A, O>, &'static str> {
        Ok(AgentLoop {
            provider: self.provider.ok_or("provider is required")?,
            store: self.store.ok_or("store is required")?,
            approver: self.approver.ok_or("approver is required")?,
            output: self.output.ok_or("output is required")?,
            registry: self.registry,
            role_dna: self.role_dna.ok_or("role_dna is required")?,
            session_id: self.session_id,
            messages: Vec::new(),
        })
    }
}

impl<P, S, A, O> Default for AgentLoopBuilder<P, S, A, O> {
    fn default() -> Self {
        Self::new()
    }
}
