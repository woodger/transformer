# OpenSearch training telemetry deployment

> Тип: руководство по развёртыванию. Provider-owned metrics v7 projection for
> Training Telemetry Query v3.

OpenSearch is not a model registry and is never called directly by Inventory
or Terminal. Transformer writes v7 point/run projections and validates them
before serving normalized telemetry reports.

The examples use `OPENSEARCH_ENDPOINT` and a mode-0600
`OPENSEARCH_NETRC` credential file. Keep it outside the repository and remove
or rotate it according to deployment secret policy.

## Flight v15 clean cut

Metrics v6 indices are incompatible with the v15 runtime. Do this only after
all Transformer services using the same deployment have stopped and after the
operator has decided that historical telemetry may be deleted.

First inspect exact targets:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v6?v"
```

If the output confirms only the expected old indices, delete those explicit
names and no wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v6,metrics-runs-v6"
```

This deletion is irreversible. It is not performed by PostgreSQL migration
0027 or by the service at startup.

## Install v7 templates before index creation

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-points-v7" \
  --data-binary \
  @app/contracts/metrics/v7/opensearch/metrics-points-v7.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-runs-v7" \
  --data-binary \
  @app/contracts/metrics/fit_run/v7/opensearch/metrics-runs-v7.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-points-v7"

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-runs-v7"
```

Templates must exist before either index is created. Installing a template does
not repair an already dynamically mapped index. The Transformer service account
needs only bulk-create, `_mget`, and bounded `_search` access to these indices;
template and index deletion permissions are administrative.

## Configure and verify

Configure the endpoint and service credential through deployment secret
configuration. Do not print, commit, or pass the password in a command line
that may be retained in shell history. Start the service only after templates
and indices are ready.

After a completed new v15 fit, verify that `metrics-runs-v7` contains its
terminal completion marker and that `metrics-points-v7` contains all expected
epoch observations. Query the public Training Telemetry v3 action for Consumer
behavior; do not make index names or mappings part of Consumer code.
