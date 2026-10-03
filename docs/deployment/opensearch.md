# Развёртывание training telemetry OpenSearch

> Тип: руководство по развёртыванию. Принадлежащая provider-у projection
> metrics v11 для Training Telemetry Query v4.

OpenSearch не является registry моделей и никогда не вызывается напрямую
Inventory или Terminal. Transformer записывает projections point/run v11 и
валидирует их до выдачи нормализованных reports telemetry.

В примерах используются `OPENSEARCH_ENDPOINT` и credential file
`OPENSEARCH_NETRC` с mode 0600. Храните его вне repository и удаляйте либо
ротируйте согласно policy secret deployment.

## Конфигурация публикатора

Задайте настройки сервиса в project `.env` или окружении процесса. Приоритет
источников описан в
[операционном руководстве Flight](../operations/flight-service.md#конфигурация-окружения).
`OPENSEARCH_NETRC` используется только административными командами `curl`;
сервис получает credentials из `OPENSEARCH_USERNAME` и
`OPENSEARCH_PASSWORD`.

| Переменная | Назначение |
| --- | --- |
| `OPENSEARCH_ENDPOINT` | Обязательный HTTP(S) origin без path, query, fragment или credentials в URL |
| `OPENSEARCH_DEPLOYMENT_ID` | Обязательная стабильная identity deployment для записи и чтения проекции |
| `OPENSEARCH_USERNAME` | Пользователь публикатора |
| `OPENSEARCH_PASSWORD` | Пароль публикатора |
| `OPENSEARCH_CA_FILE` | Путь к существующему файлу доверенного CA для HTTPS |

`OPENSEARCH_DEPLOYMENT_ID` содержит от 1 до 128 символов ASCII: первый —
буква или цифра, остальные — буквы, цифры, `.`, `_` или `-`.
Для HTTPS обязательны username, password и CA file; username `admin`
отклоняется без учёта регистра. В доверенной локальной HTTP-сети credentials
можно опустить либо задать полной парой; CA file для HTTP не допускается.
Точные правила проверки находятся в
[`opensearch/config.py`](../../app/service/adapters/outbound/opensearch/config.py).

Если ни одна из перечисленных переменных не задана, OpenSearch отключён.
Некорректная конфигурация отключает доставку и отражается в событии
`metrics.publisher.disabled` с `reason: invalid-configuration`; fit, predict
и публикация модели продолжают работать независимо от telemetry.

## Текущий runtime Semantic v5 / Flight v22

Индексы metrics v10 несовместимы с projection metrics v11. Выполняйте эту
процедуру только после остановки всех сервисов Transformer, использующих один deployment, и
после решения оператора, что historical telemetry можно удалить.

Сначала проверьте точные targets:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v10?v"
```

Если output подтверждает только ожидаемые старые индексы, удалите эти явные
имена и не используйте wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v10,metrics-runs-v10"
```

Это удаление необратимо. Оно не выполняется migration PostgreSQL 0029 или
сервисом при запуске.

## Установить templates v11 до создания индексов

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-points-v11" \
  --data-binary \
  @app/contracts/metrics/v11/opensearch/metrics-points-v11.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-runs-v11" \
  --data-binary \
  @app/contracts/metrics/fit_run/v11/opensearch/metrics-runs-v11.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-points-v11"

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-runs-v11"
```

Templates должны существовать до создания любого из индексов. Установка template
не исправляет уже динамически отображённый index. Учётной записи сервиса
Transformer нужны только доступы bulk-create, `_mget` и bounded `_search` к
этим индексам; permissions на удаление template и index являются
административными.

После завершённого нового fit v22 проверьте, что `metrics-runs-v11` содержит
terminal completion marker, а `metrics-points-v11` — все ожидаемые observations
epoch. Для поведения вызывающей системы запрашивайте публичный action Training
Telemetry v4; не делайте имена index или mappings частью её кода.
