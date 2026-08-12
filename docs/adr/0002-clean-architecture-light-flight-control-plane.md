# ADR 0002: Clean Architecture-light для control plane Flight

- Статус: принято
- Дата: 2026-07-23

## Контекст

Flight control plane вырос вокруг рабочих и проверенных границ:
`TransformerFlightServer`, `JobCoordinator`, `Ledger`, `UploadHandler`,
`OutputHandler` и `WorkerPool`. Нормативный Flight v1 contract, PostgreSQL
ledger, runtime spool и запуск существующих `fit-stream`/`predict-stream`
subprocess уже задают production-семантику сервиса.

По мере развития сервиса несколько модулей стали одновременно отвечать за
оркестрацию, инфраструктурный IO и lifecycle. В частности, upload объединяет
чтение Flight stream с durable staging, а worker — планирование очередей,
формирование доверенного CLI-вызова, управление subprocess, публикацию
артефактов и завершение attempt. Универсальные dictionary records из ledger
также делают внутренние контракты неявными.

Полный переход к классической Clean Architecture потребовал бы новой глобальной
иерархии packages, repository-интерфейсов и dependency-injection wiring. Это не
устраняет текущие риски само по себе и создало бы широкое изменение вокруг уже
стабильных внешних и durable contracts.

## Решение

Flight control plane рефакторится постепенно в стиле Clean
Architecture-light внутри существующей package-oriented архитектуры.
Выделяются локальные ответственности и явные внутренние records, но не
создаются глобальные слои `domain/application/infrastructure`.

На всём протяжении рефакторинга сохраняются:

- нормативный Flight v1 contract, action names, descriptors, Arrow schemas и
  error mapping;
- существующие module-level facades и их runtime wiring;
- PostgreSQL как единственный durable control-plane source of truth, текущая
  schema и транзакционные гарантии;
- ownership и layout runtime spool, storage epoch и startup reconciliation;
- ownership и атомарная публикация checkpoint и prediction artifacts;
- существующие CLI-команды, trusted argv, framed protocol и отдельный
  subprocess с `shell=False`;
- state machine, revision, idempotency, recovery, cancellation race и device
  semantics.

Torch execution не переносится в Flight RPC handler или в service process.
Изменение внутренней структуры само по себе не является основанием менять
Flight, persistence, spool или CLI contract.

## Последовательность выделения компонентов

Рефакторинг выполняется следующими независимыми шагами:

1. **Upload session.** Отделить состояние одного DoPut, чтение metadata и
   RecordBatch, временный IPC writer и cleanup от тонкого `UploadHandler`.
   Durable rename и ledger commit остаются единственной границей публикации:
   один DoPut по-прежнему создаёт один semantic frame.
2. **Trusted execution plan.** Отделить построение immutable server-controlled
   плана запуска от очереди и subprocess IO. План формируется только из
   валидированной job configuration и server-owned paths; сетевой запрос не
   получает способ передать path или произвольный CLI argument.
3. **Subprocess lifecycle.** Отделить spawn, process-group identity,
   параллельный stdout/stderr IO, timeout и TERM/KILL escalation от scheduler.
   `WorkerPool` продолжает быть стабильным facade для queue notification,
   capacity и shutdown.
4. **Artifact publication.** Отделить validation и staging prediction output,
   а также checkpoint publication, от subprocess supervision. Успех job
   фиксируется только после атомарной публикации всех требуемых артефактов;
   failed или cancelled attempt ничего не публикует.
5. **Attempt lifecycle.** Собрать claim, active-attempt tracking, cancel
   notification, terminal outcome и cancel-vs-success policy вокруг одного
   явного lifecycle. Автоматический retry прерванного `RUNNING` fit не
   добавляется.
6. **Typed persistence records and state policies.** Заменить пересекающие
   границы неформализованные dictionaries на минимальные immutable records и
   явные ORM mappers. Чистые transition decisions размещаются рядом с текущей
   Flight state policy; revision increments и race arbitration остаются внутри
   PostgreSQL transactions.
7. **Coordinator and ledger slices.** После стабилизации предыдущих seams
   разделить крупные модули по существующим use cases: create/status,
   seal/start/cancel, upload/input, attempt/execution, output/model и
   maintenance/recovery. Существующие facades продолжают делегировать этим
   slices, поэтому bootstrap и callers не мигрируют одновременно.

Порядок является частью решения: сначала изолируются IO и execution seams,
затем типизируются их данные, и только после этого дробятся coordinator и
ledger. Один extraction step не должен одновременно менять observable
behavior другого.

## Дисциплина изменений

Каждый commit должен быть самостоятельно deployable и revertible:

- сначала фиксируется текущее observable behavior тестами;
- facade переключается на одну новую реализацию в том же commit;
- старый и новый путь не выполняют одну mutation параллельно;
- незавершённая последующая стадия не требуется для запуска предыдущей;
- contract, recovery и failure-injection tests проходят после каждого шага.

Dual-write состояния, idempotency records, spool artifacts, outputs или models
запрещён. Для каждой mutation существует ровно один authoritative execution
path. Временное чтение старого durable представления допустимо только как
явная backward compatibility и не создаёт второй источник истины.

Новые generic packages (`common`, `shared`, глобальные
`domain/application/infrastructure`), DI container и универсальный repository
layer не вводятся. Интерфейс или локальный adapter добавляется только вместе с
реальным consumer и конкретной инфраструктурной границей; он не должен
зеркалировать целиком существующий `Ledger` или `Spool`.

## Последствия

- Риск рефакторинга ограничивается одной ответственностью и одним change set.
- Flight clients, operators и существующие CLI workflows не требуют
  синхронной миграции.
- Durable ownership и crash consistency остаются проверяемыми на каждой
  промежуточной версии.
- `WorkerPool`, `Ledger` и coordinator уменьшаются постепенно, без массового
  переноса файлов и переписывания production wiring.
- В течение миграции facades могут временно оставаться крупнее желаемого и
  делегировать новым внутренним компонентам.
- Локальные records и policies могут иметь единственную production
  реализацию; это сознательный выбор в пользу явного контракта без
  преждевременной взаимозаменяемости.

## Отклонённые альтернативы

- **Полная Clean Architecture миграция одним изменением.** Слишком широкий
  blast radius для contract-, persistence- и process-sensitive сервиса.
- **Новая parallel implementation с dual-write.** Создаёт неоднозначный source
  of truth и новые crash windows.
- **Сначала разделить ledger по таблицам.** Закрепляет persistence model вместо
  use-case boundaries и преждевременно размножает repository abstractions.
- **Перенести training/prediction in-process.** Нарушает принятую subprocess
  boundary и меняет cancellation, isolation и retry semantics.
