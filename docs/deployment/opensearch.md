# Развёртывание training telemetry OpenSearch

> Тип: руководство по развёртыванию. Принадлежащая provider-у projection
> metrics v7 для Training Telemetry Query v3.

OpenSearch не является registry моделей и никогда не вызывается напрямую
Inventory или Terminal. Transformer записывает projections point/run v7 и
валидирует их до выдачи нормализованных reports telemetry.

В примерах используются `OPENSEARCH_ENDPOINT` и credential file
`OPENSEARCH_NETRC` с mode 0600. Храните его вне repository и удаляйте либо
ротируйте согласно policy secret deployment.

## Исторический чистый переход Flight v15

Индексы metrics v6 несовместимы с runtime v15. Выполняйте это только после
остановки всех сервисов Transformer, использующих один deployment, и после
решения оператора, что historical telemetry можно удалить.

Сначала проверьте точные targets:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v6?v"
```

Если output подтверждает только ожидаемые старые индексы, удалите эти явные
имена и не используйте wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v6,metrics-runs-v6"
```

Это удаление необратимо. Оно не выполняется migration PostgreSQL 0027 или
сервисом при запуске.

## Установить templates v7 до создания индексов

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

Templates должны существовать до создания любого из индексов. Установка template
не исправляет уже динамически отображённый index. Учётной записи сервиса
Transformer нужны только доступы bulk-create, `_mget` и bounded `_search` к
этим индексам; permissions на удаление template и index являются
административными.

## Настроить и проверить

Настройте endpoint и credential service через secret configuration deployment.
Не печатайте, не коммитьте и не передавайте password в command line, который
может сохраниться в shell history. Запускайте service только после подготовки
templates и indices.

После завершённого нового fit v15 проверьте, что `metrics-runs-v7` содержит
terminal completion marker, а `metrics-points-v7` — все ожидаемые observations
epoch. Для поведения вызывающей системы запрашивайте публичный action Training
Telemetry v3; не делайте имена index или mappings частью её кода.
