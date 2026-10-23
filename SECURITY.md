# Security policy

## Report a vulnerability privately

Do not report a vulnerability in a public issue or pull request. GitHub private
vulnerability reporting is enabled for this repository. Open the repository's
**Security** tab, choose **Report a vulnerability**, and submit the report there:

https://github.com/cortexkit/synapse/security/advisories/new

Include the affected release or full source commit, platform, reproduction steps,
expected security boundary, and observed impact. Provide the smallest safe
reproducer; do not include credentials, private model data or unrelated user
content. If sensitive evidence is needed, describe it first rather than publishing
it in a public discussion.

## Scope and coordinated disclosure

Reports about model/artifact integrity, worker isolation, IPC authentication,
tampering with persisted jobs or approvals, or dependency vulnerabilities
are welcome. Identify the actual Synapse version and configuration so maintainers
can assess applicability. This policy does not promise support for every
historical version or a fixed response timeline.

Coordinate publication through the private advisory discussion. Ordinary bugs
without a security impact follow [CONTRIBUTING.md](CONTRIBUTING.md).
