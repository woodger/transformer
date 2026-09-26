# Нефункциональные требования

> Тип: политика. Cross-cutting acceptance criteria для изменений Transformer.

Этот документ используется как review checklist и не дублирует текущую
топологию, wire schemas, configuration defaults или runbook. Их источники:

- [архитектура](../architecture.md) — процессы и data ownership;
- [Контракт Flight v21](../../app/contracts/flight/v21/README.md) — wire и
  lifecycle semantics;
- [runtime обучения](../training-runtime.md) — checkpoint, recovery и обучение;
- [аутентификация](../authentication.md) — credential verification и security
  boundary;
- [Операционное руководство Flight](../operations/flight-service.md) — storage,
  devices, startup, recovery и shutdown;
- [политика metrics](./metrics-policy.md) — best-effort observability.

## Корректность и воспроизводимость

- Заданные seed и deterministic mode сохраняют заявленную семантику.
- Transport payload и RecordBatch boundaries не меняют training trajectory.
- Optimizer, scheduler, early stopping и best checkpoint принадлежат целому
  training job, а не отдельному transport payload.
- Явно запрошенный `gpu` не подменяется CPU.
- Shape, dtype, non-finite values и masking semantics валидируются до
  вычисления.

## Контракты

- Arrow schema, logical dataset identity и protocol versioning меняются только
  намеренно вместе с нормативным contract.
- Binary stdout не смешивается с diagnostics.
- Checkpoint metadata достаточно для совместимого prediction; несовместимый
  или повреждённый checkpoint не интерпретируется эвристически.
- Published model generation неизменяема и появляется только после успешного
  fit.
- Idempotency и retry не создают повторный Torch execution для одной
  зафиксированной операции.

## Надёжность и ресурсы

- Durable control-plane state и filesystem artifacts сохраняют ownership,
  определённый архитектурой и Operations Guide.
- Потеря transient storage не выдаётся за успешное recovery persistent fit.
- Fit возобновляется только с зарегистрированной границы полной global epoch;
  повреждённый recovery не приводит к silent restart.
- File publication и связанный database transition не оставляют ссылку на
  частично записанный artifact.
- Queues, caches, retry, telemetry и worker resources имеют явные bounds.
- Ошибка best-effort telemetry не меняет fit, model/job lifecycle, startup или
  shutdown.
- Потерянное GPU device не возвращается в scheduler до безопасной границы,
  определённой Operations Guide.

## Безопасность и эксплуатация

- Каждый Flight transport аутентифицирован; owner isolation соответствует
  credential model.
- TLS/plaintext выбирается только deployment configuration и не меняет
  прикладное состояние.
- Network request не управляет произвольными filesystem paths или subprocess
  arguments.
- Hot paths и idle loops не создают неограниченный I/O или polling.
- SIGTERM, cancellation и process-group cleanup не оставляют активных workers.
- CLI, environment names, logging и exit behavior рассматриваются как
  публичное поведение и меняются намеренно.

Изменение, которое выдаёт правильный happy-path результат, но нарушает одно из
этих свойств, не считается корректным.
