//! Message and event types for the agent harness.

use serde::{Deserialize, Serialize};

/// Role for a message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    /// For tool messages, the ID of the tool call this is responding to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For assistant messages, any tool calls made.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCallInfo>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

/// Information about a tool call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// A tool call from the provider stream.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// Usage statistics from a provider response.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Events emitted by the provider during streaming.
#[derive(Debug, Clone)]
pub enum ProviderEvent {
    /// A chunk of text from the assistant.
    TextDelta(String),
    /// A tool call requested by the assistant.
    ToolCall(ToolCall),
    /// The response is complete.
    Done { usage: Usage },
    /// An error occurred.
    Error(String),
}

/// Tool specification (name, description, JSON Schema parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Provider-neutral request to send to an LLM.
/// System prompt is in its own field, separate from messages.
#[derive(Debug, Clone)]
pub struct Request {
    /// The system prompt (separate from messages).
    pub system_prompt: String,
    /// The conversation messages (user, assistant, tool).
    pub messages: Vec<Message>,
    /// Tool specifications for the model to use.
    pub tools: Vec<ToolSpec>,
}

impl Request {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }
}

/// Audit record for every tool call outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub agent_id: String,
    pub role: String,
    pub tool: String,
    pub args: serde_json::Value,
    pub tier: Option<u8>,
    pub decision: String,
    pub reason: Option<String>,
    pub approver: Option<String>,
    pub outcome: String,
}

impl AuditRecord {
    pub fn new(
        agent_id: impl Into<String>,
        role: impl Into<String>,
        tool: impl Into<String>,
        args: serde_json::Value,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            role: role.into(),
            tool: tool.into(),
            args,
            tier: None,
            decision: String::new(),
            reason: None,
            approver: None,
            outcome: String::new(),
        }
    }

    pub fn with_tier(mut self, tier: u8) -> Self {
        self.tier = Some(tier);
        self
    }

    pub fn with_decision(mut self, decision: impl Into<String>) -> Self {
        self.decision = decision.into();
        self
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    pub fn with_approver(mut self, approver: impl Into<String>) -> Self {
        self.approver = Some(approver.into());
        self
    }

    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = outcome.into();
        self
    }
}

/// Session entry for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub agent_id: String,
    pub role: String,
    pub timestamp: i64,
    pub msg_role: MessageRole,
    pub payload: serde_json::Value,
}
