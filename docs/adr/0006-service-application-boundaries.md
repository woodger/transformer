# ADR 0006: прикладные границы Flight service

- Статус: принято
- Дата: 2026-08-12
- Уточняет: service boundaries из ADR 0004

## Контекст

После перехода на Flight v3 процессы service, worker и admin уже имели
раздельные composition roots и корректное направление импортов. Однако
control-plane use cases всё ещё получали конкретный PostgreSQL ledger,
управляли его транзакционными методами и формировали Flight response documents.
Inbound adapters одновременно координировали ledger, artifact storage и часть
job lifecycle. Формальный import graph был направлен внутрь, но прикладная
граница оставалась неполной.

Flight v3 является текущим штатным публичным контрактом. Решение не вводит
Flight v2 dispatcher, aliases, fallback или иной v2 compatibility surface.

## Решение

Service разделяется по ответственности:

```text
Flight request
    → inbound validation и mapping
    → application command/query
    → capability port
    → PostgreSQL/artifact adapter
    → application result
    → Flight presentation
```

Application принимает типизированные команды и запросы и возвращает
нейтральные результаты. Action names, descriptor paths, Arrow schema IDs,
camelCase wire fields и кодирование JSON принадлежат inbound Flight adapter.

PostgreSQL adapter владеет:

- транзакциями и row locks;
- idempotency records и проверкой client-generated job identity;
- fencing checks непосредственно перед mutation commit;
- преобразованием строк и page projections PostgreSQL в domain/application
  records;
- декодированием ранее записанного результата Flight v3 при точном replay.

Последний пункт сохраняет уже зафиксированные v3 idempotency results во время
внутреннего рефакторинга. Он не является поддержкой старой версии протокола.

Composition roots `service/bootstrap/job_control.py` и
`service/bootstrap/data_plane.py` являются единственными местами сборки этих
use cases и adapters. Inbound adapters не импортируют outbound implementations
и не получают raw ledger.

## Граница DoPut

Inbound adapter владеет тем, что неотделимо от Flight transport:

- чтением `FlightStreamReader` и единственного application metadata document;
- проверкой canonical Arrow physical schema и значений RecordBatch;
- созданием Arrow IPC candidate и выдачей одного `PutResult`.

Application lifecycle владеет разрешением upload:

- состоянием входа и соответствием operation;
- external fence;
- выбором устройства и storage class;
- reserve, commit, replay и abort через capability port.

Artifact storage передаётся inbound adapter через application-owned port.
Durable commit остаётся раньше единственного `PutResult`; повторная проверка
fence выполняется перед commit. Неизвестный исход PostgreSQL commit по-прежнему
сохраняет final artifact для reconciliation.

## Граница DoGet

Application query проверяет terminal state, разрешает output и выпускает либо
проверяет ticket через port. Inbound adapter владеет Flight descriptor,
`FlightInfo`, memory-mapped Arrow stream, cancellation mapping и transport
metrics. Он не обращается к ledger напрямую.

## Worker как осознанное исключение

Worker остаётся изолированным процессом и цельным runtime-модулем. Его
application executor может напрямую зависеть от PyArrow, Torch и filesystem,
поскольку эти зависимости составляют принадлежащий worker data path, а не
нарушают service boundary. Дополнительные worker ports/adapters не вводятся без
измеримой проблемы, второго runtime implementation или отдельной задачи.

## Compatibility facades

`app.flight` временно сохраняет старые Python import paths и конструкторы для
внутренних consumers и тестов. Эти файлы только делегируют сборку текущим
composition roots и не содержат job lifecycle или protocol dispatcher.
Standalone file/stream CLI остаётся действующим локальным интерфейсом и не
является альтернативной реализацией Flight job lifecycle.

## Последствия

- application use cases можно проверять без Flight, PyArrow, SQLAlchemy и
  filesystem;
- transport presentation и durable transaction semantics имеют разных
  владельцев;
- PostgreSQL race guarantees остаются внутри одного outbound adapter;
- upload/output adapters сохраняют только transport orchestration;
- worker не подвергается широкому рефакторингу без доказанной необходимости;
- AST-тесты запрещают Flight contracts в service application и concrete
  outbound dependencies во inbound adapters.
