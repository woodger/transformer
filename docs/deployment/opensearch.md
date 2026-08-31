# Доставка training metrics в OpenSearch

> Тип: руководство по развёртыванию. Настройка OpenSearch projection
> Transformer.

Best-effort boundary и ownership telemetry описаны в
[`политике metrics`](../policy/metrics-policy.md). Текущие schemas и templates
находятся в
[`app/contracts/metrics/v4`](../../app/contracts/metrics/v4/README.md) и
[`app/contracts/metrics/fit_run/v3`](../../app/contracts/metrics/fit_run/v3/README.md).

Текущее развёртывание использует доверенную локальную сеть:

```text
http://hp260g9.home:9200
```

REST TLS отключён, OpenSearch требует существующую Basic Auth.

## Подготовить templates и индексы

Операцию выполняет администратор OpenSearch до включения publisher-а.

Обычный index и data stream не могут одновременно использовать одно имя. Если
в кластере уже существуют прежние indices или data streams
`metrics-points-v3`, `metrics-runs-v2`, сначала остановите publisher и
отдельно решите вопрос
сохранения их данных. Эта инструкция намеренно ничего не удаляет.

```bash
search_endpoint=http://hp260g9.home:9200
read -r -s -p 'OpenSearch password: ' OPENSEARCH_PASSWORD
printf '\n'

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-points-v4" \
  --data-binary \
  @app/contracts/metrics/v4/opensearch/metrics-points-v4.template.json

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-runs-v3" \
  --data-binary \
  @app/contracts/metrics/fit_run/v3/opensearch/metrics-runs-v3.template.json

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --request PUT \
  "$search_endpoint/metrics-points-v4"

curl --fail --silent --show-error \
  --user "admin:$OPENSEARCH_PASSWORD" \
  --request PUT \
  "$search_endpoint/metrics-runs-v3"

unset OPENSEARCH_PASSWORD
```

Templates закрепляют `dynamic: strict` и `number_of_replicas: 0`. Не
преобразуйте индексы в data streams и не назначайте им rollover alias или ISM
rollover policy: проверка повторного `create` и `_mget` требует одного concrete
index на каждую versioned projection.

Transformer публикует только projection `inventory.metrics.v5` в текущие
versioned indices `metrics-points-v4` и `metrics-runs-v3`. Поддержки прежних
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
выключенным; service продолжает работать.

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
