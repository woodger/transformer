# ADR 0003: долговечное возобновляемое обучение и device-aware execution

- Статус: принято
- Дата: 2026-07-24
- Заменяет: решения о recovery, runtime storage и CUDA scheduling из ADR 0001

## Контекст

Flight v1 намеренно связывал jobs и Arrow inputs с временной storage epoch
`/tmp/transformer`. Поэтому restart сервиса или хоста завершал активный fit
ошибкой, даже если обучение уже выполнялось много часов. Worker также
предоставлял одну логическую CUDA lane и не привязывал claimed attempt к
конкретному физическому GPU.

Training path уже владеет едиными для всех payload-ов optimizer, loss schedule,
checkpoint selector и lifecycle early stopping. Это состояние можно сохранить
на границе завершённой global epoch, не загружая весь dataset в память.

## Решение

Flight v2 заменяет v1. Runtime compatibility с actions, descriptors и job
records v1 не предоставляется.

Committed fit inputs и внутренние training-recovery checkpoints находятся в
постоянном каталоге project `recovery/`. PostgreSQL остаётся авторитетным
источником их видимости и состояния job, attempt и retry. Файл, не
зарегистрированный атомарной операцией Ledger, является orphan. PostgreSQL не
хранит Arrow payload-ы или blobs checkpoints.

Первый recovery format сохраняет состояние только после полной global epoch.
Он содержит текущие model, optimizer, AMP scaler, training progress,
early-stopping state, выбор best checkpoint и состояние random generator.
Поэтому при resume теряется не более незавершённой epoch. Recovery checkpoints
являются внутренними artifacts и никогда не получают `modelRef`; только
успешный fit публикует неизменяемую модель в `models/`.

Прерванный fit или fit с потерянным device закрывает активную attempt и
переходит в `RETRYING`. Следующая attempt запускается с последнего корректного
checkpoint, зарегистрированного в PostgreSQL, либо с нулевой epoch, если ни
один checkpoint не регистрировался. Отсутствующий, повреждённый или
несовместимый зарегистрированный checkpoint приводит к явной recovery error,
а не разрешает незаметно перезапустить обучение.

CUDA scheduling использует привязанный к boot inventory физических devices.
CUDA attempt получает принадлежащий серверу lease одного device, а её
subprocess привязывается через `CUDA_VISIBLE_DEVICES`. Подтверждённая потеря
device помещает его в quarantine до конца текущей загрузки. Running process не
переносится между devices: он завершается, и создаётся новая attempt. Fit
остаётся в `RETRYING`, если не осталось исправных CUDA devices, и может
продолжиться после reboot, заново сформировавшего inventory.

Retry разрешён только после прерывания service/host и подтверждённой потери
device. Cancellation, invalid input, несовместимые recovery data, CUDA
out-of-memory, обычная ошибка subprocess, malformed output и заполнение disk не
приводят к автоматическому retry.

## Структура хранилища

```text
<project-root>/recovery/
  jobs/{jobId}/
    inputs/{ordinal}.arrow
    checkpoints/{generation}.pth

/tmp/transformer/
  service.lock
  storage-epoch
  cuda-quarantine.json
  spool/jobs/{jobId}/attempts/{attempt}/

<project-root>/models/
  {modelRef}/
    checkpoint.pth
    metadata.json
```

Постоянные inputs читаются напрямую. Начальным механизмом ускорения RAM служит
page cache операционной системы; v2 не добавляет вторую копию tmpfs или
протокол cache coherency.

## Публичный контракт

Flight v2 добавляет состояние `RETRYING`, безопасный recovery progress,
динамическую CUDA capacity и количество devices в quarantine. Paths
filesystem, arguments worker-а и физические идентификаторы GPU наружу не
передаются. Fit всегда допускает resume и не имеет управляемого клиентом
переключателя recovery.

## Что не входит в решение

- runtime compatibility с Flight v1;
- resume в середине epoch или отдельного batch;
- распределение одной job между несколькими GPU;
- универсальная abstraction storage или repository;
- инициализация CUDA в процессе Flight service;
- автоматический CPU fallback для CUDA attempt;
- хранение payload-ов или checkpoints в PostgreSQL;
- возврат quarantined GPU в service до reboot.

## Последствия

- Полная потеря `/tmp/transformer` больше не делает недействительными
  возобновляемые fit jobs с корректными постоянными inputs и
  зарегистрированным checkpoint.
- Recovery storage требует явного мониторинга capacity и очистки.
- Файлы checkpoints становятся больше, поскольку recovery включает optimizer,
  best model и random state.
- Resume на другом GPU сохраняет логическое training state, но не обещает
  побитового равенства для nondeterministic CUDA kernels.
- Переключение на v2 делает недействительными runtime jobs, attempts, tickets и
  idempotency records v1, сохраняя опубликованные models, aliases и access
  tokens.
