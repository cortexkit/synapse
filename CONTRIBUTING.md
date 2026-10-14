# Contributing

Thanks for looking at Synapse. This page says how a change gets in, and what a change has to prove before it can.

## Open an issue first

**Every change starts as an issue, and a pull request follows only after a maintainer says go.** Label it `bug` for something that is broken, or `design discussion` for anything that adds, removes, or changes behaviour.

Changes can invalidate lane certification, rotate a vector-space fingerprint and require re-embedding, or change worker recovery on a platform not exercised by hosted CI. Agreeing on those consequences before implementation avoids a patch that solves the local symptom but breaks a consumer boundary.

A good issue says what is wrong or missing, how you noticed, and what you would do about it. If you already have a patch, say so — that is useful information, not a reason to skip the step.

Issues that report a defect with evidence are welcome without any further ceremony, including ones you do not intend to fix yourself.

## Then the pull request

The go is a label: a maintainer adds `design-approved` to the issue once the shape is agreed. Put the issue number on the `Approved issue: #` line of the pull request template. `Refs #N` also works, but a closing keyword is never required, because maintainers close an issue when the fix ships, not when it merges. A contribution gate checks this automatically. A pull request whose linked issue is not yet approved is turned into a draft, and it becomes ready for review on its own once the label is applied. When there is genuinely no design to agree, such as a typo or a broken link, a maintainer can label the pull request `trivial` instead.

Keep the pull request to the issue's scope; a second improvement noticed along the way is a second issue.

What a review checks:

- **It builds where we cannot see it.** CI runs Linux and Windows. The macOS Metal, CUDA, and Vulkan lanes are manual gates a maintainer dispatches — say if your change touches them so we run the right one.
- **The tests fail without the fix.** We mutation-check: a test that passes on unpatched code proves nothing, and we will say so.
- **No new suppressions.** No `as any`-equivalents: `#[allow(...)]` added to quiet a real lint, `unsafe` without a stated invariant, an ignored test, or a widened tolerance to make a gate pass.
- **Comments explain the reason,** for a reader who was not in the discussion. No references to issue numbers, review rounds, or "as discussed".
- **`Cargo.lock` at `origin/master` bytes.** Leave the lock as it is unless your change needs a dependency bump, and if it does, say so in the issue. The subc crates move together at exact versions, so a bump to one of them usually means bumping the set.

Claims in a PR description get verified against the source, and numbers get re-measured where we can. That is not distrust — it is the same standard we hold our own changes to, and it is why a merged change here can be relied on.

## Verify the change

Run from the repository root with the committed lockfile:

```sh
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo check --locked -p synapse-module --features test-support
cargo test --locked -p synapse-module --features test-support --lib --test it --test skeleton_e2e --test wire_contract
cargo fmt --all --check
bash scripts/public-banlist-check.sh
bash scripts/check-cortexkit-layout.sh
```

Use the relevant package and evidence gates in
[development verification](docs/development.md) and
[backend certification](docs/backend-certification.md). Explain platform or
checkpoint skips explicitly; a portable pass is not hardware certification.
Public documentation belongs in flat topic files under `docs/`; release evidence
consumed by code stays tracked at its declared paths. See
[third-party notices](THIRD-PARTY-NOTICES.md) before importing adapted source.

## Security

Use the private reporting procedure in [SECURITY.md](SECURITY.md); do not open a public issue for a vulnerability.

## License

Contributions are accepted under the repository's [MIT license](LICENSE).
