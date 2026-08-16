# Доставка training metrics в OpenSearch

> Type: Operations. Настройка OpenSearch projection Transformer.

Источник истины и границы решения описаны в
[ADR 0009](../adr/0009-centralized-training-metrics.md), нормативные schemas и
templates — в
[`app/contracts/metrics/v1`](../../app/contracts/metrics/v1/README.md).

Текущее развёртывание использует доверенную локальную сеть:

```text
http://hp260g9.home:9200
```

TLS, authentication, пользователи и роли для него не настраиваются.

## Подготовить templates и индексы

Операцию выполняет администратор OpenSearch до включения publisher-а.

Обычный index и data stream не могут одновременно использовать одно имя. Если
в кластере уже существуют data streams `metrics-points-v1` или
`metrics-artifacts-v1`, сначала остановите publisher и отдельно решите вопрос
сохранения их данных. Эта инструкция намеренно ничего не удаляет.

```bash
search_endpoint=http://hp260g9.home:9200

curl --fail --silent --show-error \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-points-v1" \
  --data-binary \
  @app/contracts/metrics/v1/opensearch/metrics-points-v1.template.json

curl --fail --silent --show-error \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-artifacts-v1" \
  --data-binary \
  @app/contracts/metrics/v1/opensearch/metrics-artifacts-v1.template.json

curl --fail --silent --show-error \
  --request PUT \
  "$search_endpoint/metrics-points-v1"

curl --fail --silent --show-error \
  --request PUT \
  "$search_endpoint/metrics-artifacts-v1"
```

Templates закрепляют `dynamic: strict` и `number_of_replicas: 0`. Не
преобразуйте индексы в data streams и не назначайте им rollover alias или ISM
rollover policy: проверка повторного `create` и `_mget` требует одного concrete
index на каждую versioned projection.

## Настроить Transformer

Добавьте в project `.env`:

```dotenv
OPENSEARCH_ENDPOINT=http://hp260g9.home:9200
OPENSEARCH_DEPLOYMENT_ID=hp800g9.home
```

Для HTTP-профиля не задавайте `OPENSEARCH_USERNAME`, `OPENSEARCH_PASSWORD` и
`OPENSEARCH_CA_FILE`. `deploymentId` различает установки Transformer в общей
платформе и участвует в semantic identity каждого point.

Версия приложения и Git commit записываются в каждый artifact и point. При
развёртывании из Git checkout commit определяется автоматически. Если каталог
`.git` не поставляется вместе с приложением, дополнительно задайте полный
lowercase SHA-1 развёрнутого commit:

```dotenv
TRANSFORMER_GIT_COMMIT=0123456789abcdef0123456789abcdef01234567
```

Если ни одной `OPENSEARCH_*` переменной нет, publisher выключен, но model
publication продолжает создавать durable artifact и outbox backlog. Частичная
или смешанная конфигурация считается ошибкой deployment и не позволяет
запустить service.

После изменения `.env` перезапустите service. Migrations применяются отдельно:

```bash
./.venv/bin/python ./app/main.py db migrations status
./.venv/bin/python ./app/main.py db migrations apply
```

Текущий head — `0008`.

## Проверить работу

После короткого fit проверьте:

- модель содержит `models/{modelRef}/metrics.jsonl`;
- health показывает gauges `metricsOutboxEntries`, `metricsOutboxBytes` и
  `metricsOutboxOldestAgeSeconds`;
- журнал содержит `metrics.artifact.delivered`;
- поиск по `runId`, `transformerJobId` или `modelRef` возвращает points и один
  artifact document;
- повторная доставка не создаёт второй документ с тем же `_id`.

`metrics.delivery.retry_scheduled` означает временную ошибку.
`metrics.delivery.blocked` означает schema/mapping/integrity error: такая entry
автоматически не повторяется, пока причина не диагностирована.
