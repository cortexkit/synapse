#![deny(unsafe_op_in_unsafe_fn)]

//! Read-only native sources whose bindings require unsafe calls.
//! Formatting, architecture policy and injectable source seams live in synapse-core.

#[cfg(target_os = "windows")]
pub mod windows;
