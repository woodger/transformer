# Эксплуатация сервиса Transformer Arrow Flight

> Тип: операционное руководство. Запуск, shutdown, storage и recovery текущего
> сервиса Flight v15.

Wire semantics определены [Flight v15](../../app/contracts/flight/v15/README.md).
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
Catalog или shutdown. Она делает telemetry query unavailable либо
unavailable-after-terminal в соответствии с его query contract.

## CPU budget CUDA Worker-а

| Переменная окружения | По умолчанию | Назначение |
| --- | --- | --- |
| `TRANSFORMER_CUDA_TORCH_INTRAOP_THREADS` | `8` | Число PyTorch intra-op threads одного CUDA Worker-а |
| `TRANSFORMER_CUDA_TORCH_INTEROP_THREADS` | `1` | Число PyTorch inter-op threads одного CUDA Worker-а |

Defaults принадлежат `app/config.py`. Service передаёт проверенные значения
только CUDA Worker-процессу, а Worker применяет их один раз до загрузки model и
training runtime. CPU attempts сохраняют собственные PyTorch defaults. Thread
budget является operational configuration и не входит во Flight, Worker,
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

## Релиз Flight v15

Migration 0027 удаляет предыдущее state boundary. Остановите все instances
service, дождитесь terminal state jobs, примените её, замените индексы metrics
OpenSearch на v7, затем deploy service v15. Старые models, checkpoints, state
recovery и telemetry после этого использовать нельзя. См.
[управление migrations](database-migrations.md) и
[deployment OpenSearch](../deployment/opensearch.md).

## Health и troubleshooting

Используйте `transformer.v15.health` для аутентифицированной surface health
provider-а и `transformer.v15.capabilities` для текущей availability
device/upload/query. Для операционной диагностики используйте logs service и
state database. Никогда не помещайте bearer credentials, passwords database или
raw paths checkpoint-а в общие logs или сообщения support.
