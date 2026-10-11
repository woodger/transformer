# Настройка OpenSearch для показателей обучения

> Тип: руководство по развёртыванию. Подключение OpenSearch и создание индексов метрик.

Transformer отправляет в OpenSearch метрики эпох и итоги обучения. Затем он
проверяет эти данные и возвращает клиенту отчёты через свой API. Inventory
и Terminal не обращаются к OpenSearch напрямую. Каталог моделей хранится
в PostgreSQL.

В примерах используются `OPENSEARCH_ENDPOINT` и файл учётных данных
`OPENSEARCH_NETRC` с правами 0600. Храните его вне репозитория и удаляйте или
обновляйте по правилам хранения секретов вашей установки.

## Конфигурация публикатора

Задайте настройки сервиса в project `.env` или окружении процесса. Приоритет
источников описан в
[операционном руководстве Flight](../operations/flight-service.md#конфигурация-окружения).
`OPENSEARCH_NETRC` используется только административными командами `curl`;
сервис получает учётные данные из `OPENSEARCH_USERNAME` и
`OPENSEARCH_PASSWORD`.

| Переменная | Назначение |
| --- | --- |
| `OPENSEARCH_ENDPOINT` | Обязательный HTTP(S) адрес сервера без пути, параметров, фрагмента или учётных данных в URL |
| `OPENSEARCH_DEPLOYMENT_ID` | Обязательный постоянный идентификатор установки для записи и чтения метрик |
| `OPENSEARCH_USERNAME` | Пользователь публикатора |
| `OPENSEARCH_PASSWORD` | Пароль публикатора |
| `OPENSEARCH_CA_FILE` | Путь к существующему файлу доверенного CA для HTTPS |

`OPENSEARCH_DEPLOYMENT_ID` содержит от 1 до 128 символов ASCII: первый —
буква или цифра, остальные — буквы, цифры, `.`, `_` или `-`.
Для HTTPS обязательны имя пользователя, пароль и файл доверенного CA.
Имя `admin` отклоняется без учёта регистра. В доверенной локальной HTTP-сети
учётные данные можно опустить либо задать полной парой. Файл CA для HTTP
не допускается.
Точные правила проверки находятся в
[`opensearch/config.py`](../../app/service/adapters/outbound/opensearch/config.py).

Если ни одна из перечисленных переменных не задана, OpenSearch отключён.
Некорректная конфигурация отключает доставку и отражается в событии
`metrics.publisher.disabled` с `reason: invalid-configuration`; fit, predict
и публикация модели продолжают работать независимо от метрик.

## Обновление существующей установки

Текущий сервис использует метрики v13, связанные с контрольными точками v14.
При переходе с Flight v23 на v24 требуется заменить прежние индексы v12
и установить шаблоны v13. Это отдельная административная операция;
сервис и миграции PostgreSQL её не выполняют.

Индексы метрик v12 несовместимы с форматом v13. Перед заменой остановите все
экземпляры Transformer этой установки и убедитесь, что прежние показатели
обучения можно удалить.

Сначала проверьте точные имена индексов:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  "$OPENSEARCH_ENDPOINT/_cat/indices/metrics-*-v12?v"
```

Если ответ содержит только ожидаемые старые индексы, удалите эти явные
имена и не используйте wildcard:

```bash
curl --fail --silent --show-error \
  --netrc-file "$OPENSEARCH_NETRC" \
  --request DELETE \
  "$OPENSEARCH_ENDPOINT/metrics-points-v12,metrics-runs-v12"
```

Это удаление необратимо. Ни миграция PostgreSQL `0031`, ни сервис при запуске
его не выполняют.

## Установить шаблоны и создать индексы

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

Шаблоны должны существовать до создания любого из индексов. Установка шаблона
не исправляет уже созданный индекс с автоматическим определением типов полей.
Учётной записи сервиса Transformer нужны только создание документов через
bulk-create, чтение через `_mget` и ограниченные по объёму выборки `_search`
из этих индексов. Удаление шаблонов и индексов требует административных прав.

После завершения нового обучения проверьте, что `metrics-runs-v13` содержит
запись о завершении, а `metrics-points-v13` — все ожидаемые метрики эпох.
Работу клиента проверяйте через API показателей обучения. Имена индексов
и устройство полей OpenSearch остаются деталями сервиса.
