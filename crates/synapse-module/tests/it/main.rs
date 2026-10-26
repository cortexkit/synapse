// Logger tests and daemon tests that install a global subscriber stay in
// separate executables so their tracing configuration cannot leak here.
#[path = "../common/mod.rs"]
mod common;
mod common_tests;
mod launch_nonce_readers;
mod launch_refusals;
mod owned_decode_acceptance;
mod worker_host_timeout;
mod worker_liveness;
