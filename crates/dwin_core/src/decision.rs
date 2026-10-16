//! Decision enum with most-restrictive-wins logic.

use serde::{Deserialize, Serialize};

/// Decision returned by hooks for BeforeToolCall events.
/// Most restrictive wins: Deny > RequireApproval > Allow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Decision {
    /// Allow the tool call to proceed.
    #[default]
    Allow,
    /// Deny the tool call with a reason.
    Deny(String),
    /// Require human approval with a reason.
    RequireApproval(String),
}

impl Decision {
    /// Returns the restrictiveness level (higher = more restrictive).
    /// Deny = 2, RequireApproval = 1, Allow = 0.
    pub fn restrictiveness(&self) -> u8 {
        match self {
            Decision::Allow => 0,
            Decision::RequireApproval(_) => 1,
            Decision::Deny(_) => 2,
        }
    }

    /// Combine two decisions, returning the most restrictive one.
    pub fn most_restrictive(self, other: Decision) -> Decision {
        if self.restrictiveness() >= other.restrictiveness() {
            self
        } else {
            other
        }
    }

    /// Combine multiple decisions, returning the most restrictive one.
    /// If empty, returns Allow.
    pub fn combine<I: IntoIterator<Item = Decision>>(decisions: I) -> Decision {
        decisions
            .into_iter()
            .fold(Decision::Allow, |acc, d| acc.most_restrictive(d))
    }

    /// Returns true if this is a Deny decision.
    pub fn is_deny(&self) -> bool {
        matches!(self, Decision::Deny(_))
    }

    /// Returns true if this is a RequireApproval decision.
    pub fn is_require_approval(&self) -> bool {
        matches!(self, Decision::RequireApproval(_))
    }

    /// Returns true if this is an Allow decision.
    pub fn is_allow(&self) -> bool {
        matches!(self, Decision::Allow)
    }

    /// Get the reason if this is a Deny or RequireApproval.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Decision::Allow => None,
            Decision::Deny(reason) | Decision::RequireApproval(reason) => Some(reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restrictiveness_ordering() {
        assert!(Decision::Allow.restrictiveness() < Decision::RequireApproval("".to_string()).restrictiveness());
        assert!(Decision::RequireApproval("".to_string()).restrictiveness() < Decision::Deny("".to_string()).restrictiveness());
    }

    #[test]
    fn test_most_restrictive_deny_wins() {
        let allow = Decision::Allow;
        let require = Decision::RequireApproval("need approval".to_string());
        let deny = Decision::Deny("blocked".to_string());

        let result = Decision::combine([allow.clone(), require.clone(), deny.clone()]);
        assert!(result.is_deny());
    }

    #[test]
    fn test_most_restrictive_require_approval_wins_over_allow() {
        let allow = Decision::Allow;
        let require = Decision::RequireApproval("need approval".to_string());

        let result = Decision::combine([allow, require]);
        assert!(result.is_require_approval());
    }

    #[test]
    fn test_combine_empty_returns_allow() {
        let result = Decision::combine(std::iter::empty());
        assert!(result.is_allow());
    }
}
