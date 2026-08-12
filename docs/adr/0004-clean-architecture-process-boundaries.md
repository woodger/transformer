# ADR 0004: границы процессов Clean Architecture

- Статус: принято
- Дата: 2026-08-06
- Заменяет: решения о package boundaries из ADR 0002
- Версии контрактов изменены ADR 0005

## Контекст

Transformer содержит три независимо запускаемых процесса: сервис Arrow Flight,
короткоживущий ML worker одной execution attempt и административный CLI.
Предыдущий рефакторинг в стиле Clean Architecture-light разделил долговечный
upload, планирование execution, supervision subprocess, publication artifacts,
state policies и use-case slices PostgreSQL, но оставил их внутри технического
package `app.flight`.

Flight является inbound transport, а не владельцем lifecycle job, PostgreSQL,
artifact storage, scheduling или ML execution. Между service и ML worker также
существует настоящая process boundary: import реализации worker в service
инициализировал бы или связал его с PyArrow/Torch concerns, которыми service не
владеет.

## Решение

Transformer применяет Clean Architecture независимо для каждого runtime
process.

```text
service/bootstrap -> inbound Flight + outbound adapters + application/domain
worker/bootstrap  -> worker application + Arrow/Torch/checkpoint runtime
admin/bootstrap   -> CLI + required application use cases and PostgreSQL
```

Глобального bootstrap, импортирующего все implementations, нет. Top-level CLI
dispatcher лениво выбирает один bootstrap.

Правило dependencies сервиса:

```text
Flight adapter -> service application -> service domain
outbound adapters -> application-owned capability ports
bootstrap -> all service implementations
```

Application ports называются по capabilities: `JobRepository`,
`ArtifactPublisher`, `ExecutionPlanBuilder`, `AttemptProcess`,
`WorkerExecutor`, `WorkerCapabilities` и `DeviceLeaseManager`. Они не
называются в честь PostgreSQL, filesystems или CUDA. Handle running process не
экспортируется, поскольку `AttemptProcess` владеет child от spawn до
окончательного reap. Generic repository, facade unit of work и DI container не
вводятся.

Service не импортирует реализацию worker. Оба процесса могут зависеть от
нейтрального независимо версионируемого `app.contracts.worker.v1`. Публичный
контракт Flight v2 и внутренний контракт worker v1 развиваются независимо.

## Identity и владение attempt

Каждый claim в PostgreSQL увеличивает публичный числовой attempt и создаёт
новый UUID `attemptId` в той же transaction. Одна attempt запускает не более
одного worker subprocess, а владение не передаётся, пока её `attemptId` активен.
Recovery сервиса закрывает прерванную attempt вместо повторного подключения к
её процессу. Разрешённый retry получает новые ordinal и `attemptId`.

Поэтому `attemptId` является equality fence. Application mutations атомарно
проверяют job, identity активной attempt и точные states, разрешённые для
мутации. Второй random token дублировал бы этот lifecycle. Монотонная fence
epoch откладывается до появления design, допускающего передачу владения без
создания новой attempt.

## Протокол процесса и artifacts

Один worker process обрабатывает одну execution attempt. Принадлежащий service
неизменяемый JSON manifest передаёт configuration и управляемые paths. Arrow
IPC используется только для объёмных input/output artifacts; ограниченные
версионируемые JSON events передают lifecycle и progress; stderr содержит
diagnostics; process exit является только фактом завершения transport.

Worker пишет только в workspace своей attempt. Он никогда не публикует
публичный output, recovery generation, неизменяемую model generation или
`modelRef`. Publication выполняется поэтапно:

```text
worker temporary write
-> close/fsync
-> atomic rename inside attempt workspace
-> event with size and digest
-> service validation
-> service-owned immutable publication and fsync
-> PostgreSQL reference/state transaction
```

PostgreSQL не может атомарно включать состояние filesystem или subprocess.
После сбоя может остаться unreferenced staged или immutable artifact. Такие
orphans удаляются reconciliation. PostgreSQL никогда не должен ссылаться на
незавершённый artifact.

## Владение

| Ресурс | Владелец |
| --- | --- |
| состояние job, revision и idempotency result | PostgreSQL/service |
| ordinal активной attempt и `attemptId` | PostgreSQL/service |
| handle процесса и pipes | adapter worker process |
| committed Arrow inputs | artifact storage сервиса |
| временные checkpoint и result files | workspace attempt/worker |
| публикация долговечного recovery checkpoint | service |
| публикация immutable output/model | service |
| training и inference | worker |
| публичный Flight response | inbound adapter сервиса |

## Что не входит в решение

- изменение публичного поведения Flight v2 в рамках package migration;
- import или initialization Torch/CUDA в service process;
- постоянный worker pool до подтверждения необходимости profiling-ом;
- доступ worker к PostgreSQL, bearer authentication или публичному lifecycle
  job;
- transaction abstraction, притворяющаяся, что PostgreSQL включает состояние
  filesystem;
- ports вокруг Torch tensors и другие abstractions без реального consumer;
- dual-write или параллельные authoritative implementations во время
  migration.

## Последствия

- каждый process имеет ограниченный import graph и владеет только своими
  resources;
- worker crashes и CUDA memory изолированы одной attempt;
- service restart не доверяет поздним events старого equality fence;
- publication и reconciliation artifacts становятся явным application
  lifecycle, а не скрытыми side effects filesystem;
- migration обязана сохранять arbitration races PostgreSQL и выполнять по
  одному deployable изменению boundary за раз.
