//! Embeds the git commit this binary was built from, so `ck provenance synapse`
//! can say which source is running.
//!
//! Two compile-time variables, both optional: `SYNAPSE_BUILD_REV` (the
//! 40-character HEAD commit) and `SYNAPSE_BUILD_TREE` (`clean` or `dirty`).
//! Both stay unset when git is unavailable (a source tarball, a vendored
//! checkout), and the module then declares the commit absent with a reason
//! instead of a placeholder that would compare equal between two unidentified
//! builds.

fn main() {
    synapse_build_stamp::emit_git_provenance();
}
