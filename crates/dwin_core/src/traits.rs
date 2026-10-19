//! Core traits: Provider, Tool, Store, Hook, Approver.

use std::pin::Pin;
use std::future::Future;

use async_trait::async_trait;

use crate::{
    AuditRecord, Decision, ProviderEvent, Request, SessionEntry, ToolCall, ToolOutput, ToolSpec,
};

/// A stream of provider events.
pub type EventStream = Pin<Box<dyn futures_like::Stream<Item = ProviderEvent> + Send>>;

/// Minimal Stream trait for use without pulling in futures crate.
/// This allows us to avoid the futures dependency while still having streaming.
pub mod futures_like {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    pub trait Stream {
        type Item;
        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
    }

    impl<S: ?Sized + Stream + Unpin> Stream for &mut S {
        type Item = S::Item;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut **self).poll_next(cx)
        }
    }

    impl<S: ?Sized + Stream + Unpin> Stream for Box<S> {
        type Item = S::Item;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut **self).poll_next(cx)
        }
    }

    impl<S: ?Sized + Stream> Stream for Pin<Box<S>> {
        type Item = S::Item;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            // SAFETY: We're projecting Pin<&mut Pin<Box<S>>> to Pin<&mut S>.
            // This is safe because:
            // 1. Box provides a stable address, so the inner S won't move
            // 2. We're not exposing the &mut S, only the Pin<&mut S>
            unsafe {
                let pin_box: &mut Pin<Box<S>> = self.as_mut().get_unchecked_mut();
                let inner: Pin<&mut S> = pin_box.as_mut();
                inner.poll_next(cx)
            }
        }
    }
}

/// LLM provider trait.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stream a response from the provider.
    async fn stream(&self, req: Request) -> anyhow::Result<EventStream>;
}

/// Tool trait for implementing tools.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool specification (name, description, JSON Schema params).
    fn spec(&self) -> ToolSpec;

    /// Execute the tool with the given arguments.
    async fn call(&self, args: serde_json::Value) -> anyhow::Result<ToolOutput>;
}

/// Store trait for session persistence and audit logging.
#[async_trait]
pub trait Store: Send + Sync {
    /// Append an entry to a session.
    async fn append(&self, session_id: &str, entry: SessionEntry) -> anyhow::Result<()>;

    /// Load all entries for a session.
    async fn load(&self, session_id: &str) -> anyhow::Result<Vec<SessionEntry>>;

    /// List all session IDs.
    async fn list(&self) -> anyhow::Result<Vec<String>>;

    /// Record an audit entry. Every tool call must produce exactly one audit record.
    async fn audit(&self, record: AuditRecord) -> anyhow::Result<()>;
}

/// Hook event types for Phase 1.
#[derive(Debug, Clone)]
pub enum HookEvent<'a> {
    /// Session is starting.
    SessionStart {
        session_id: &'a str,
        agent_id: &'a str,
        role: &'a str,
    },
    /// Before sending a request to the provider.
    BeforeRequest {
        request: &'a Request,
    },
    /// Before executing a tool call.
    BeforeToolCall {
        call: &'a ToolCall,
        tier: Option<u8>,
    },
    /// After executing a tool call.
    AfterToolCall {
        call: &'a ToolCall,
        output: &'a ToolOutput,
    },
}

/// Mutable context for hooks.
pub struct HookContext<'a> {
    /// The system prompt (can be modified by hooks).
    pub system_prompt: &'a mut String,
    /// The agent ID.
    pub agent_id: &'a str,
    /// The role.
    pub role: &'a str,
}

/// Outcome from a hook.
#[derive(Debug, Clone)]
pub enum HookOutcome {
    /// Continue processing (for non-BeforeToolCall events).
    Continue,
    /// Decision for BeforeToolCall events.
    Decision(Decision),
}

impl HookOutcome {
    pub fn allow() -> Self {
        HookOutcome::Decision(Decision::Allow)
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        HookOutcome::Decision(Decision::Deny(reason.into()))
    }

    pub fn require_approval(reason: impl Into<String>) -> Self {
        HookOutcome::Decision(Decision::RequireApproval(reason.into()))
    }

    pub fn into_decision(self) -> Option<Decision> {
        match self {
            HookOutcome::Continue => None,
            HookOutcome::Decision(d) => Some(d),
        }
    }
}

/// Hook trait for intercepting events.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Handle an event. For BeforeToolCall, return a Decision via HookOutcome::Decision.
    /// For other events, return HookOutcome::Continue.
    async fn on<'a>(&self, event: &HookEvent<'a>, ctx: &mut HookContext<'a>) -> anyhow::Result<HookOutcome>;
}

/// Approval result from an Approver.
#[derive(Debug, Clone)]
pub struct Approval {
    pub approved: bool,
    pub approver: String,
}

impl Approval {
    pub fn approved(approver: impl Into<String>) -> Self {
        Self {
            approved: true,
            approver: approver.into(),
        }
    }

    pub fn denied(approver: impl Into<String>) -> Self {
        Self {
            approved: false,
            approver: approver.into(),
        }
    }
}

/// Approver trait for human approval of tool calls.
#[async_trait]
pub trait Approver: Send + Sync {
    /// Ask for approval of a tool call.
    async fn ask(&self, call: &ToolCall, reason: &str) -> anyhow::Result<Approval>;
}

/// Extension trait for async iteration over EventStream.
pub trait EventStreamExt {
    fn next(&mut self) -> impl Future<Output = Option<ProviderEvent>> + Send;
}

impl EventStreamExt for EventStream {
    fn next(&mut self) -> impl Future<Output = Option<ProviderEvent>> + Send {
        use std::task::Poll;

        struct NextFuture<'a> {
            stream: &'a mut EventStream,
        }

        impl Future for NextFuture<'_> {
            type Output = Option<ProviderEvent>;

            fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
                self.stream.as_mut().poll_next(cx)
            }
        }

        NextFuture { stream: self }
    }
}
