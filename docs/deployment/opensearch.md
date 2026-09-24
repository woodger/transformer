# Развёртывание training telemetry OpenSearch

> Тип: руководство по развёртыванию. Принадлежащая provider-у projection
> metrics v8 для Training Telemetry Query v4.

OpenSearch не является registry моделей и никогда не вызывается напрямую
Inventory или Terminal. Transformer записывает projections point/run v8 и
валидирует их до выдачи нормализованных reports telemetry.

В примерах используются `OPENSEARCH_ENDPOINT` и credential file
`OPENSEARCH_NETRC` с mode 0600. Храните его вне repository и удаляйте либо
ротируйте согласно policy secret deployment.

## Текущий чистый переход Semantic v4 / Flight v17

Индексы metrics v7 несовместимы с runtime v17. Выполняйте эту процедуру только
после остановки всех сервисов Transformer, использующих один deployment, и
после решения оператора, что historical telemetry можно удалить.

Сначала проверьте точные targets:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v7?v"
```

Если output подтверждает только ожидаемые старые индексы, удалите эти явные
имена и не используйте wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v7,metrics-runs-v7"
```

Это удаление необратимо. Оно не выполняется migration PostgreSQL 0028 или
сервисом при запуске.

## Установить templates v8 до создания индексов

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-points-v8" \
  --data-binary \
  @app/contracts/metrics/v8/opensearch/metrics-points-v8.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-runs-v8" \
  --data-binary \
  @app/contracts/metrics/fit_run/v8/opensearch/metrics-runs-v8.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-points-v8"

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-runs-v8"
```

Templates должны существовать до создания любого из индексов. Установка template
не исправляет уже динамически отображённый index. Учётной записи сервиса
Transformer нужны только доступы bulk-create, `_mget` и bounded `_search` к
этим индексам; permissions на удаление template и index являются
административными.

После завершённого нового fit v17 проверьте, что `metrics-runs-v8` содержит
terminal completion marker, а `metrics-points-v8` — все ожидаемые observations
epoch. Для поведения вызывающей системы запрашивайте публичный action Training
Telemetry v4; не делайте имена index или mappings частью её кода.
