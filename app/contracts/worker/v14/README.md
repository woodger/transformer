# Transformer Worker v14 process contract

> CONTRACT DOCUMENT. This is a provider-internal service-to-worker process
> protocol. It is not a Consumer wire contract.

Worker v14 receives a service-controlled command manifest, emits framed events,
and reads/writes only service-controlled artifact paths. The launcher accepts
only `--contract-version=14`; it has no reader or alias for earlier Worker
versions.

The manifest carries validated semantic v3 intent, its issued digests, the
resolved internal model configuration, data geometry, training/diagnostics
configuration, and checkpoint/recovery fences. Worker materializes the
Transformer-owned architecture and private resources from that information.
Opaque target identities remain data for layout and telemetry; no execution
path may branch on a Consumer target name.

The internal formats are:

```text
transformer-worker protocol 14
transformer-checkpoint-v8
transformer-recovery-v8
```

Recovery verifies the embedded checkpoint metadata, semantic identities,
resolved job configuration, input manifest, progress, and managed artifact
before state is loaded. It does not remap targets, weights, or prediction
coordinates across definitions.

The Worker consumes the same logical indexed-feature-block input layout as
Flight v15. It derives block positions, reconstructs bounded slices into the
logical Float32 tensor, and produces target-aligned prediction vectors. The
physical Arrow schema identifiers remain provider-owned process details.

Worker v14 is activated only by the Flight v15 clean cut. Existing Worker
artifacts cannot be resumed or converted.
