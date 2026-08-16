# Доставка training metrics в OpenSearch

> Type: Operations. Настройка OpenSearch projection Transformer.

Источник истины и границы решения описаны в
[ADR 0009](../adr/0009-centralized-training-metrics.md), нормативные schemas и
templates — в
[`app/contracts/metrics/v1`](../../app/contracts/metrics/v1/README.md).

## Подготовить templates и индексы

Операцию выполняет администратор OpenSearch. Transformer service не должен
получать эти права.

Обычный index и data stream не могут одновременно использовать одно имя. Если
в кластере уже существуют data streams `metrics-points-v1` или
`metrics-artifacts-v1`, сначала остановите publisher и отдельно решите вопрос
сохранения их данных. Эта инструкция намеренно не удаляет существующие
streams или backing indices.

```bash
search_endpoint=https://hp260g9.home:9200
search_ca=/path/to/opensearch-ca.pem

curl --fail --silent --show-error \
  --cacert "$search_ca" \
  --user admin \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-points-v1" \
  --data-binary \
  @app/contracts/metrics/v1/opensearch/metrics-points-v1.template.json

curl --fail --silent --show-error \
  --cacert "$search_ca" \
  --user admin \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$search_endpoint/_index_template/metrics-artifacts-v1" \
  --data-binary \
  @app/contracts/metrics/v1/opensearch/metrics-artifacts-v1.template.json

curl --fail --silent --show-error \
  --cacert "$search_ca" \
  --user admin \
  --request PUT \
  "$search_endpoint/metrics-points-v1"

curl --fail --silent --show-error \
  --cacert "$search_ca" \
  --user admin \
  --request PUT \
  "$search_endpoint/metrics-artifacts-v1"
```

`--user admin` запрашивает password интерактивно. Не передавайте password в
аргументе команды. После создания проверьте, что templates имеют
`dynamic: strict`, а оба индекса используют ожидаемые mappings.

Не преобразуйте индексы в data streams и не назначайте им rollover alias или
ISM rollover policy. Глобальная уникальность `_id` и проверка через `_mget`
требуют одного concrete index на каждую versioned projection. Причина и
условие снятия ограничения зафиксированы в ADR 0009.

## Создать отдельного writer-а

Создайте пользователя, например `transformer-metrics`, и роль только для
patterns `metrics-points-v1` и `metrics-artifacts-v1`. Для фактических Bulk
create и `_mget` нужны минимальные действия:

```text
cluster:
  indices:data/write/bulk
  indices:data/read/mget

index:
  indices:data/write/bulk*
  indices:data/write/index*
  indices:data/read/mget*
  indices:admin/resolve/index
```

Проверьте набор на установленном Security plugin representative Bulk и `_mget`
requests. Writer не должен иметь `delete`, `update`, index creation, template,
mapping, data stream, ISM или cluster-admin permissions. Пользователь `admin`
для runtime запрещён самим Transformer.

## Настроить Transformer

Добавьте в project `.env`:

```dotenv
OPENSEARCH_ENDPOINT=https://hp260g9.home:9200
OPENSEARCH_USERNAME=transformer-metrics
OPENSEARCH_PASSWORD=replace-with-a-secret
OPENSEARCH_CA_FILE=/path/to/opensearch-ca.pem
OPENSEARCH_DEPLOYMENT_ID=hp800g9.home
```

Все пять параметров задаются вместе. `OPENSEARCH_ENDPOINT` обязан быть HTTPS
origin без path, CA verification нельзя отключить. `deploymentId` различает
несколько установок Transformer в общей платформе и участвует в semantic
identity каждого point.

Версия приложения и Git commit записываются в каждый artifact и point. При
развёртывании из Git checkout commit определяется автоматически. Если каталог
`.git` не поставляется вместе с приложением, дополнительно задайте полный
lowercase SHA-1 развёрнутого commit:

```dotenv
TRANSFORMER_GIT_COMMIT=0123456789abcdef0123456789abcdef01234567
```

Если ни одной переменной нет, publisher выключен, но model publication всё
равно создаёт durable artifact и outbox backlog. Частичная конфигурация
считается ошибкой deployment и не позволяет запустить service.

После изменения `.env` перезапустите service. Migrations применяются отдельно:

```bash
./.venv/bin/python ./app/main.py db migrations status
./.venv/bin/python ./app/main.py db migrations apply
```

Текущий head — `0007`.

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
