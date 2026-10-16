//! Built-in hooks including the DateHook example.

use async_trait::async_trait;
use chrono::Utc;

use crate::{Hook, HookContext, HookEvent, HookOutcome};

/// Hook that appends the current date to the system prompt.
/// This is the example hook required by the spec.
pub struct DateHook;

impl DateHook {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DateHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for DateHook {
    async fn on<'a>(&self, event: &HookEvent<'a>, ctx: &mut HookContext<'a>) -> anyhow::Result<HookOutcome> {
        if let HookEvent::BeforeRequest { .. } = event {
            let date = Utc::now().format("%Y-%m-%d").to_string();
            if !ctx.system_prompt.contains(&format!("Current date: {}", date)) {
                ctx.system_prompt.push_str(&format!("\n\nCurrent date: {}", date));
            }
        }
        Ok(HookOutcome::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_date_hook_appends_date() {
        let hook = DateHook::new();
        let mut system_prompt = "You are an assistant.".to_string();
        let mut ctx = HookContext {
            system_prompt: &mut system_prompt,
            agent_id: "test",
            role: "test",
        };
        let request = crate::Request::new("");
        let event = HookEvent::BeforeRequest { request: &request };

        hook.on(&event, &mut ctx).await.unwrap();

        let today = Utc::now().format("%Y-%m-%d").to_string();
        assert!(ctx.system_prompt.contains(&format!("Current date: {}", today)));
    }

    #[tokio::test]
    async fn test_date_hook_idempotent() {
        let hook = DateHook::new();
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let mut system_prompt = format!("You are an assistant.\n\nCurrent date: {}", today);
        let original = system_prompt.clone();
        let mut ctx = HookContext {
            system_prompt: &mut system_prompt,
            agent_id: "test",
            role: "test",
        };
        let request = crate::Request::new("");
        let event = HookEvent::BeforeRequest { request: &request };

        hook.on(&event, &mut ctx).await.unwrap();

        assert_eq!(*ctx.system_prompt, original);
    }
}
