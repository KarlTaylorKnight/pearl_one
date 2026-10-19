//! Registry for tools, hooks, and other components.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{Hook, Tool, ToolSpec};

/// Registry for tools and hooks.
#[derive(Default)]
pub struct Registry {
    tools: HashMap<String, Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn Hook>>,
    /// Default risk tiers for tools.
    risk_tiers: HashMap<String, u8>,
}

impl Registry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), Arc::new(tool));
    }

    /// Register a tool with an Arc.
    pub fn register_tool_arc(&mut self, tool: Arc<dyn Tool>) {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), tool);
    }

    /// Get a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Get all tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get tool specs for the given allowlist.
    /// Only returns specs for tools that are both registered and in the allowlist.
    pub fn get_tool_specs(&self, allowlist: &[String]) -> Vec<ToolSpec> {
        allowlist
            .iter()
            .filter_map(|name| self.tools.get(name).map(|t| t.spec()))
            .collect()
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Register a hook.
    pub fn register_hook<H: Hook + 'static>(&mut self, hook: H) {
        self.hooks.push(Arc::new(hook));
    }

    /// Register a hook with an Arc.
    pub fn register_hook_arc(&mut self, hook: Arc<dyn Hook>) {
        self.hooks.push(hook);
    }

    /// Get all hooks.
    pub fn hooks(&self) -> &[Arc<dyn Hook>] {
        &self.hooks
    }

    /// Set the default risk tier for a tool.
    pub fn set_risk_tier(&mut self, tool: impl Into<String>, tier: u8) {
        self.risk_tiers.insert(tool.into(), tier);
    }

    /// Get the default risk tier for a tool.
    pub fn get_risk_tier(&self, tool: &str) -> Option<u8> {
        self.risk_tiers.get(tool).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolOutput, ToolSpec};
    use async_trait::async_trait;

    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: format!("Mock tool: {}", self.name),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: serde_json::Value) -> anyhow::Result<ToolOutput> {
            Ok(ToolOutput::success("mock result"))
        }
    }

    #[test]
    fn test_registry_tools() {
        let mut registry = Registry::new();
        registry.register_tool(MockTool { name: "read".to_string() });
        registry.register_tool(MockTool { name: "write".to_string() });

        assert!(registry.has_tool("read"));
        assert!(registry.has_tool("write"));
        assert!(!registry.has_tool("bash"));

        let specs = registry.get_tool_specs(&["read".to_string(), "bash".to_string()]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "read");
    }
}
