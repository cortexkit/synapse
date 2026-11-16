# Findings-changed v1 samples

These fixtures show the notice and read-only `events_get` replies a Basal flow
sees. Only `findings_changed` is declared as a flow-trigger event. The notice has
an empty body, the UUID in `Nats-Msg-Id` and `event_key`, and the stored body's
bare lowercase SHA-256 in `Ck-Content-Digest`. `box_test` stands for the bus account.

The body returned by `events_get` is the same stored JSON string before and after
a later findings generation. The `events_get` reply's RFC 3339 `published_at` is
the event's store commit time, not the publisher's sent marker. Wrong keys, names, and versions
return `event_not_visible`.

Regenerate with:

```sh
SYNAPSE_REGENERATE_EVENT_SAMPLES=1 cargo test --locked -p synapse-module --features test-support --lib event_outbox::tests::findings_changed_samples_v1_are_current -- --exact
```
The same test without the environment variable fails when any sample or
`SHA256SUMS` is stale. The checksums cover the JSON samples, not this README.
