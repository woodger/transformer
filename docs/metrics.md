# Метрики и OpenSearch

> Type: Policy. Документ фиксирует best-effort границу observability и
> запрещает влияние telemetry на прикладное поведение Transformer.

OpenSearch используется только как механизм наблюдаемости и не должен влиять на поведение приложения.

Rationale best-effort artifact/outbox boundary сохранён в
[ADR 0009](./adr/0009-centralized-training-metrics.md). Текущие нормативные
правила задаёт этот документ.

* Метрики не должны влиять на бизнес-логику, управление потоком выполнения, корректность работы или поведение API.
* Ошибки сбора или отправки метрик не должны приводить к ошибке основной операции.
* Не использовать OpenSearch как состояние приложения, хранилище, механизм синхронизации или источник истины.
* Не добавлять новые метрики без необходимости, связанной с текущей задачей, или явной эксплуатационной ценности.
* Предпочитать небольшой набор стабильных и полезных метрик избыточной детализации.

### Проектирование метрик

* Для одинаковых понятий использовать согласованные имена, единицы измерения и семантику полей во всех проектах.
* Использовать только ограниченные и предсказуемые наборы значений измерений.
* Не использовать `requestId`, идентификаторы инструментов, временные метки, тексты ошибок, stack trace и другие высококардинальные значения как измерения метрик без явной необходимости.
* Не записывать в метрики секреты, токены, учетные данные и чувствительные данные.
* Не дублировать в метриках информацию, которая уже полноценно представлена в логах.

### Границы инструментации

* OpenSearch-зависимый код должен оставаться внутри существующего слоя метрик или инфраструктуры.
* Domain и основная application-логика не должны напрямую зависеть от OpenSearch.
* Предпочитать инструментирование существующих границ выполнения, а не перестраивать код приложения ради сбора метрик.
* Не расширять сбор метрик на несвязанные компоненты без предварительного согласования изменения области задачи.

### Структурные владельцы

Telemetry образует отдельный наблюдающий slice и не является частью ML-state
или lifecycle модели:

```text
worker/training/epoch.py                 core результат global epoch
worker/telemetry/                        необязательные runtime-наблюдения
service/application/telemetry/           records и delivery orchestration
service/application/ports/telemetry.py   capability boundaries
service/adapters/outbound/artifacts/telemetry/
service/adapters/outbound/postgres/telemetry/
service/adapters/outbound/opensearch/
```

Service domain и общий PostgreSQL ledger не владеют telemetry records или
metrics API. Core model publisher не строит metrics artifacts. После
прикладной публикации модели отдельный telemetry publisher может создать
run-owned artifacts и outbox с identity `jobId`.

Job progress содержит только checkpoint-aligned core-поля `epoch`, `step`,
`loss_stage` и `loss`. Они фиксируются атомарно с recovery checkpoint и
возвращаются через Flight status. AMP, gradient, per-target metrics, timings и
полный epoch document не являются состоянием job. Поля `completedEpochs` и
`globalStep` остаются в `recovery.latestCheckpoint` как fallback.
Административный lifecycle моделей не знает о доставке OpenSearch. Удаление
модели не отменяет и не удаляет telemetry run.

Target-bound telemetry использует одну регистрозависимую semantic identity из
`inventory.target.v2`: структурную пару `target.index`/`target.name` с именами
`MeanReturn`, `SigmaReturn`, `ProbTP`, `ProbSL`, `VolatilityNext` и
`HittingProbTP`. Старые camelCase формы и numeric enum ordinal Consumer-а не
являются contract values. Нормативную identity задаёт текущий
[`Flight v5 contract`](../app/contracts/flight/v5/README.md).

### Надёжность

Метрики являются вспомогательными observability-данными и собираются по принципу best effort.

Сбор метрик должен иметь ограниченное потребление ресурсов и не создавать неограниченные очереди, бесконечные retry, рост памяти или существенную задержку основной операции.

Недоступность OpenSearch или инфраструктуры метрик не должна нарушать корректность работы приложения.

### Граница фиксации

Прикладные checkpoint, model generation и terminal state фиксируются без
зависимости от telemetry. Epoch metrics сохраняются отдельной best-effort
операцией после recovery checkpoint, а run-owned metrics artifacts и outbox —
после публикации модели.

Отсутствующая, повреждённая или отброшенная telemetry не должна:

* переводить fit в `FAILED`;
* препятствовать `SUCCEEDED` и публикации модели;
* блокировать удаление модели;
* блокировать запуск или остановку сервиса.

Некорректная конфигурация publisher-а отключает только доставку метрик и
фиксируется в operational logs/counters. Outbox имеет ограничение по числу
записей и объёму, конечный retry budget и срок хранения terminal entries.
После terminal retention metadata и файлы run-owned telemetry удаляются вместе.

Transformer не вычисляет статистики значений training targets для Consumer и
не проектирует metadata локального artifact в отдельный OpenSearch index.
Централизованная проекция состоит из epoch points и terminal run summary.

### Согласованность между проектами

Если Node.js и Python проекты измеряют одну и ту же операцию или понятие, следует стремиться к совместимым именам метрик, единицам измерения, семантике полей и структуре документов.

Не изменять общий контракт метрик только в одном проекте, если это приведёт к расхождению эквивалентных метрик между компонентами.
