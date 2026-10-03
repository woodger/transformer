# Эксплуатация сервиса Transformer Arrow Flight

> Тип: операционное руководство. Запуск, shutdown, storage и recovery текущего
> сервиса Flight v22.

Wire semantics определены [Flight v22](../../app/contracts/flight/v22/README.md).
Этот документ описывает эксплуатацию сервиса, а не JSON details вызывающей
системы.

## Требования runtime

- `.venv` проекта и зафиксированные зависимости Python;
- PostgreSQL, доступный с host-а service;
- persistent настроенные directories runtime, models, recovery и telemetry;
- окружение процесса Linux, способное завершать принадлежащие service groups
  process Worker;
- `nvidia-smi`, только когда требуется scheduling GPU;
- OpenSearch, только когда Training Telemetry Query должна materialize reports.

Недоступность OpenSearch не блокирует fit, predict, публикацию model, Model
Catalog, Model Topology или shutdown. Она делает telemetry query unavailable
либо unavailable-after-terminal в соответствии с его query contract.

## Конфигурация окружения

Подготовьте `.env` в корне проекта по
[`.env.example`](../../.env.example), заменив примерные значения настройками
своего deployment. Адаптеры PostgreSQL и OpenSearch читают этот файл без
изменения `os.environ`; одноимённые переменные окружения процесса имеют
приоритет над `.env`. Эти настройки PostgreSQL используют и Flight service,
и локальные команды `auth`, `models` и `db`.

| Переменная | Назначение |
| --- | --- |
| `POSTGRES_HOST` | Обязательный адрес PostgreSQL |
| `POSTGRES_DB` | Обязательное имя базы данных |
| `POSTGRES_USER` | Обязательный пользователь базы данных |
| `POSTGRES_PASSWORD` | Обязательный непустой пароль |
| `POSTGRES_PORT` | Необязательный порт; значение по умолчанию задано в `app/config.py` |

Схема PostgreSQL также задаётся значением по умолчанию в `app/config.py`;
отдельной переменной окружения для её выбора нет. Разбор и проверка подключения
определены в
[`postgres/config.py`](../../app/service/adapters/outbound/postgres/config.py).
Проверка и применение schema описаны в
[руководстве migrations](database-migrations.md).

Настройки публикатора и запросов OpenSearch находятся в
[руководстве OpenSearch](../deployment/opensearch.md#конфигурация-публикатора).
Для работы без OpenSearch оставьте перечисленные там переменные незаданными
как в `.env`, так и в окружении процесса.

Лимиты и таймауты `TRANSFORMER_*` читаются только из окружения процесса,
а не из `.env`. Допустимый набор и правила проверки определяет
[`bootstrap/config.py`](../../app/service/bootstrap/config.py), значения по
умолчанию — [`app/config.py`](../../app/config.py). Адрес, порт и TLS задаются
параметрами `flight serve`; `runtime_dir`, `cpu_capacity` и
`retention_seconds` используют значения по умолчанию и не читаются из
`TRANSFORMER_*`.

## CPU budget CUDA Worker-а

`app/config.py` фиксирует число PyTorch intra-op threads одного CUDA Worker-а
равным восьми, inter-op threads — одному. После validation command manifest
Worker применяет эти значения только при `device.backend = cuda`, до загрузки
model и training runtime. CPU attempts сохраняют собственные PyTorch defaults.
Thread budget является operational configuration и не входит во Flight, Worker,
checkpoint или model compatibility contracts.

## Запуск и остановка

Примените migrations до старта. В deployment используйте production unit
systemd:

```bash
sudo systemctl start transformer
systemctl status transformer
journalctl -u transformer -f
sudo systemctl stop transformer
```

Process получает exclusive locks для настроенных directories runtime и recovery.
Два instances service должны использовать разные managed storage roots; один
instance никогда не должен очищать runtime artifacts другого.

При старте service проверяет revision database, завершает только надёжно
идентифицированные orphan process groups Worker и завершает все незавершённые
jobs до запуска WorkerPool. `WAITING_INPUT`, `QUEUED`, `RUNNING` и `RETRYING`
становятся `FAILED / EXECUTION_INTERRUPTED`; `CANCELLING` становится
`CANCELLED`. Поэтому job не может автоматически продолжить выполнение после
любого restart service. Затем service удаляет файлы без references только внутри
собственных locked managed roots. Он не сканирует произвольные filesystem paths
или OpenSearch как authority.

## Владение storage

PostgreSQL авторитетен для lifecycle job, idempotency, owners, attempts,
metadata published model, API tokens и state telemetry outbox. Managed
filesystem storage содержит durable inputs, artifacts attempt/recovery,
published checkpoints и временные files telemetry. OpenSearch — best-effort
projection.

Перед фиксацией prediction outputs и `SUCCEEDED` service выполняет `fsync`
файлов и их managed parent directories, включая созданные Worker-ом каталоги.
Ошибка синхронизации не позволяет опубликовать частичный набор outputs.

Directories published model содержат только managed artifacts checkpoint
provider-а. Catalog/detail разрешает их metadata из PostgreSQL; ни filesystem
scan, ни telemetry не могут создать видимую generation модели.

## Recovery и cleanup

Recovery применяется только для retryable сбоя Worker внутри работающего
service и только из зарегистрированного checkpoint-а, у которого точно
совпадают configuration job, input manifest, semantic identities и fences
progress. После restart service recovery не запускается: job уже terminal и
для новой попытки вызывающая система создаёт новый job. Corrupt или incompatible
artifact завершается ошибкой; он никогда не переназначается молча.

Maintenance удаляет terminal artifacts job согласно настроенному retention и
выполняет запрошенное удаление model. Startup reconciliation удаляет только
artifacts без references в тех же service roots. Не удаляйте вручную rows
PostgreSQL или managed directories model, чтобы принудить cleanup; используйте
`models delete` либо документированную migration clean cut.

## Текущий runtime Semantic v5 / Flight v22

Migration 0029 удаляет state предыдущей semantic boundary. Для перехода
на Semantic v5 остановите все instances service, дождитесь terminal state jobs
и примените её. Перед запуском Flight v22 замените индексы metrics OpenSearch
v10 на v11 по documented procedure. Старые models, checkpoints, state recovery
и telemetry до migration 0029 использовать нельзя.

См. [управление migrations](database-migrations.md) и
[deployment OpenSearch](../deployment/opensearch.md).

## Health и troubleshooting

Используйте `transformer.v22.health` для аутентифицированной surface health
provider-а и `transformer.v22.capabilities` для текущей availability
device/upload/query. Для операционной диагностики используйте logs service и
state database. Никогда не помещайте bearer credentials, passwords database или
raw paths checkpoint-а в общие logs или сообщения support.
