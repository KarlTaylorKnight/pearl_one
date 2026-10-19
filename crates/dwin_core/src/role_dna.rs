//! RoleDna type representing role configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Role DNA configuration loaded from roles/<role>.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDna {
    /// Unique agent identifier (e.g., "tech.dwin@fene.ae").
    pub agent_id: String,
    /// Role name (e.g., "tech").
    pub role: String,
    /// Display name for the role (e.g., "FENE Technical Dwin").
    pub display_name: String,
    /// System prompt / job specification.
    pub system_prompt: String,
    /// Allowlist of tool names this role can use.
    pub tools: Vec<String>,
    /// Risk tier overrides per tool (optional).
    #[serde(default)]
    pub risk_tiers: HashMap<String, u8>,
}

impl RoleDna {
    /// Create a new RoleDna with the minimum required fields.
    pub fn new(
        agent_id: impl Into<String>,
        role: impl Into<String>,
        display_name: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            role: role.into(),
            display_name: display_name.into(),
            system_prompt: system_prompt.into(),
            tools: Vec::new(),
            risk_tiers: HashMap::new(),
        }
    }

    /// Add a tool to the allowlist.
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add multiple tools to the allowlist.
    pub fn with_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Set a risk tier for a tool.
    pub fn with_risk_tier(mut self, tool: impl Into<String>, tier: u8) -> Self {
        self.risk_tiers.insert(tool.into(), tier);
        self
    }

    /// Check if a tool is in the allowlist.
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        self.tools.iter().any(|t| t == tool_name)
    }

    /// Get the risk tier for a tool, if defined.
    pub fn get_risk_tier(&self, tool_name: &str) -> Option<u8> {
        self.risk_tiers.get(tool_name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_dna_creation() {
        let dna = RoleDna::new("test@fene.ae", "test", "Test Role", "You are a test.")
            .with_tools(["read", "write"])
            .with_risk_tier("write", 1);

        assert_eq!(dna.agent_id, "test@fene.ae");
        assert_eq!(dna.role, "test");
        assert!(dna.is_tool_allowed("read"));
        assert!(dna.is_tool_allowed("write"));
        assert!(!dna.is_tool_allowed("bash"));
        assert_eq!(dna.get_risk_tier("write"), Some(1));
        assert_eq!(dna.get_risk_tier("read"), None);
    }
}
