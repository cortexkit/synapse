//! Live certification tooling; synapse-module must not depend on this daemon-backed crate.
#![forbid(unsafe_code)]

pub mod command;
pub mod live;

use synapse_certify::Error;

fn refuse(message: impl Into<String>) -> Error {
    Error(message.into())
}
