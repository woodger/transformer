# Доставка и чтение training metrics в OpenSearch

> Тип: руководство по развёртыванию. Настройка OpenSearch projection
> Transformer для publisher-а и Training Telemetry Query.

Best-effort boundary и ownership telemetry описаны в
[`политике metrics`](../policy/metrics-policy.md). Текущие schemas и templates
находятся в
[`app/contracts/metrics/v6`](../../app/contracts/metrics/v6/README.md) и
[`app/contracts/metrics/fit_run/v6`](../../app/contracts/metrics/fit_run/v6/README.md).
Публичную read-only проекцию задаёт
[`Training Telemetry Query v2`](../../app/contracts/training_telemetry/v2/README.md).

Текущее развёртывание использует доверенную локальную сеть:

```text
http://hp260g9.home:9200
```

REST TLS отключён, OpenSearch требует существующую Basic Auth.

## Подготовить templates и индексы

Операцию выполняет администратор OpenSearch до включения publisher-а.

```bash
search_endpoint=http://hp260g9.home:9200
read -r -s -p 'OpenSearch password: ' OPENSEARCH_PASSWORD
printf '\n'
```

Обычный index и data stream не могут одновременно использовать одно имя.
Перед clean-cut переходом остановите publisher. Текущий runtime не читает и
не дописывает прежние `metrics-points-v5` и `metrics-runs-v5`; согласованное
удаление historical metrics выполняется отдельно после проверки точных имён:

```bash
curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  "$search_endpoint/_cat/indices/metrics-*-v5?v"

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --request DELETE \
  "$search_endpoint/metrics-points-v5,metrics-runs-v5"
```

Эти команды безвозвратно удаляют только два явно названных legacy index.
Прежние templates v5 можно удалить после переключения; они не влияют на новый
runtime.

```bash
curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-points-v6" \
  --data-binary \
  @app/contracts/metrics/v6/opensearch/metrics-points-v6.template.json

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-runs-v6" \
  --data-binary \
  @app/contracts/metrics/fit_run/v6/opensearch/metrics-runs-v6.template.json

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --request PUT \
  "$search_endpoint/metrics-points-v6"

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --request PUT \
  "$search_endpoint/metrics-runs-v6"

unset OPENSEARCH_PASSWORD
```

Templates закрепляют `dynamic: strict` и `number_of_replicas: 0`. Не
преобразуйте индексы в data streams и не назначайте им rollover alias или ISM
rollover policy: проверка повторного `create` и `_mget` требует одного concrete
index на каждую versioned projection. Service account Transformer
должен иметь доступ на bulk create, `_mget` и bounded `_search` в этих двух
индексах; administrative template/delete privileges runtime не требуются.

Transformer публикует только projection `transformer.metrics.v6` в текущие
versioned indices `metrics-points-v6` и `metrics-runs-v6`. Поддержки прежних
экспериментальных артефактов, записей outbox и индексов нет.

## Настроить Transformer

Добавьте в project `.env`:

```dotenv
OPENSEARCH_ENDPOINT=http://hp260g9.home:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=<пароль OpenSearch>
OPENSEARCH_DEPLOYMENT_ID=hp800g9.home
```

HTTP-профиль допускает либо отсутствие credentials, либо полную пару
`OPENSEARCH_USERNAME`/`OPENSEARCH_PASSWORD`; CA для него не задаётся. Текущий
deployment использует Basic Auth. `deploymentId` различает установки
Transformer в общей платформе и участвует в semantic identity каждого point.

Версия приложения и Git commit записываются в каждый artifact и point. При
развёртывании из Git checkout commit определяется автоматически. Если каталог
`.git` не поставляется вместе с приложением, дополнительно задайте полный
lowercase SHA-1 развёрнутого commit:

```dotenv
TRANSFORMER_GIT_COMMIT=0123456789abcdef0123456789abcdef01234567
```

Если ни одной `OPENSEARCH_*` переменной нет, publisher выключен, но model
publication продолжает создавать durable artifact и outbox backlog. Частичная
или смешанная конфигурация считается ошибкой deployment и оставляет publisher
выключенным; service продолжает работать. Без полной конфигурации
Training Telemetry Query возвращает structured backend-unavailable error; fit,
predict и Model Catalog от OpenSearch не зависят.

Migrations применяются отдельно по
[`операционному руководству PostgreSQL`](../operations/database-migrations.md).
Перед перезапуском service убедитесь, что `db migrations status` показывает
`Pending migrations: no`, затем примените изменение `.env` перезапуском.

## Проверить работу

После короткого fit проверьте:

- при успешном сборе telemetry run содержит
  `telemetry/{jobId}/metrics.jsonl` и
  `telemetry/{jobId}/run-summary.json`;
- health показывает gauges `metricsOutboxEntries`, `metricsOutboxBytes` и
  `metricsOutboxOldestAgeSeconds`;
- журнал содержит `metrics.run.delivered`;
- поиск по `runId`, `transformerJobId` или `modelRef` возвращает epoch points и
  один terminal fit run summary;
- повторная доставка не создаёт второй документ с тем же `_id`.
- `transformer.training-telemetry.v2.report` для owner-visible published
  model возвращает `available` только после проверки complete projection.

Перед terminal run marker publisher ожидает refresh последней партии
points, а затем refresh самого marker. Поэтому видимый marker
означает, что все ранее доставленные points уже видимы query-пути.

`metrics.delivery.retry_scheduled` означает временную ошибку.
`metrics.delivery.blocked` означает schema/mapping/integrity error: такая entry
автоматически не повторяется и удаляется после terminal retention.
`metrics.delivery.dropped` означает исчерпание retry budget;
`metrics.outbox.dropped` — отказ admission из-за заполненного outbox.

Некорректная конфигурация отключает publisher и создаёт
`metrics.publisher.disabled`, но не блокирует запуск сервиса. Одна entry
повторяется не больше 288 раз и не дольше 24 часов. Admission outbox ограничен
10 000 entries и 10 GiB; сбой или потеря telemetry не меняют fit outcome и
model lifecycle.
