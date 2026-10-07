# Развёртывание training telemetry OpenSearch

> Тип: руководство по развёртыванию. Принадлежащая provider-у projection
> metrics v13 для Training Telemetry Query v4.

OpenSearch не является registry моделей и никогда не вызывается напрямую
Inventory или Terminal. Transformer записывает projections point/run v13 и
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

## Текущий runtime Semantic v7 / Flight v24

Flight v24 использует Metrics v13, связанные с checkpoint v14. Переход с
Flight v23 требует replacement прежних индексов v12 и установки templates
v13. Ни сервис, ни migration PostgreSQL не выполняют эту операцию.

Индексы metrics v12 несовместимы с projection metrics v13. Выполняйте эту
процедуру только после остановки всех сервисов Transformer, использующих один deployment, и
после решения оператора, что historical telemetry можно удалить.

Сначала проверьте точные targets:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v12?v"
```

Если output подтверждает только ожидаемые старые индексы, удалите эти явные
имена и не используйте wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v12,metrics-runs-v12"
```

Это удаление необратимо. Оно не выполняется migration PostgreSQL 0031 или
сервисом при запуске.

## Установить templates v13 до создания индексов

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-points-v13" \
  --data-binary \
  @app/contracts/metrics/v13/opensearch/metrics-points-v13.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --header 'Content-Type: application/json' \
  --request PUT \
  "$OPENSEARCH_ENDPOINT/_index_template/metrics-runs-v13" \
  --data-binary \
  @app/contracts/metrics/fit_run/v13/opensearch/metrics-runs-v13.template.json

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-points-v13"

curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request PUT "$OPENSEARCH_ENDPOINT/metrics-runs-v13"
```

Templates должны существовать до создания любого из индексов. Установка template
не исправляет уже динамически отображённый index. Учётной записи сервиса
Transformer нужны только доступы bulk-create, `_mget` и bounded `_search` к
этим индексам; permissions на удаление template и index являются
административными.

После завершённого нового fit v24 проверьте, что `metrics-runs-v13` содержит
terminal completion marker, а `metrics-points-v13` — все ожидаемые observations
epoch. Для поведения вызывающей системы запрашивайте публичный action Training
Telemetry v4; не делайте имена index или mappings частью её кода.
