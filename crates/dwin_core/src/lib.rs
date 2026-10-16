//! Core crate for the Dwin agent harness.
//!
//! This crate contains:
//! - Message/event types
//! - Provider-neutral Request
//! - Traits: Provider, Tool, Store, Hook, Approver
//! - Decision enum with most-restrictive-wins logic
//! - RoleDna type
//! - Registry
//! - Agent loop with allowlist filtering and 25-iteration cap

mod types;
mod traits;
mod decision;
mod role_dna;
mod registry;
mod agent_loop;
mod hooks;

pub use types::*;
pub use traits::*;
pub use decision::*;
pub use role_dna::*;
pub use registry::*;
pub use agent_loop::*;
pub use hooks::*;

#[cfg(test)]
mod tests;
