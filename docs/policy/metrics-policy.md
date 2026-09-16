# Политика metrics и OpenSearch

> Тип: политика. Best-effort граница observability и запрет влияния telemetry
> на прикладное поведение Transformer.

OpenSearch используется только для наблюдаемости и не является состоянием
приложения, механизмом синхронизации или источником истины. Текущую структуру
telemetry slice описывает [архитектурный справочник](../architecture.md),
training telemetry — [справочник обучения](../training-runtime.md), нормативные
documents и projections —
[`app/contracts/metrics/`](../../app/contracts/metrics/), а настройку доставки
— [инструкция OpenSearch](../deployment/opensearch.md).
Read-only public projection полного report задаёт
[`Training Telemetry Query v3`](../../app/contracts/training_telemetry/v3/README.md).

Rationale best-effort artifact/outbox boundary сохранён в
[ADR 0009](../adr/0009-centralized-training-metrics.md).

## Прикладная граница

- Метрики не влияют на fit/predict flow control, результат операции,
  checkpoint selection, model lifecycle или model compatibility.
- Ошибка сбора, материализации, конфигурации или доставки telemetry не
  превращает основную операцию в ошибку и не блокирует startup или shutdown.
- Checkpoint, model generation и terminal state фиксируются независимо от
  telemetry.
- Удаление прикладной сущности не зависит от состояния её observability-данных.
- Public telemetry query всегда owner-gated через model registry; telemetry
  не определяет существование или доступность model generation.
- Новая метрика добавляется только при явной эксплуатационной ценности или как
  необходимая часть текущей задачи; предпочтителен небольшой стабильный набор.

## Проектирование метрик

- Для одинаковых понятий используются согласованные имена, единицы измерения и
  семантика полей.
- Dimensions имеют ограниченный и предсказуемый набор значений.
- `requestId`, timestamps, тексты ошибок, stack traces и другие
  высококардинальные значения не становятся dimensions без отдельного
  обоснования.
- Секреты, tokens, credentials и чувствительные данные в metrics не попадают.
- Метрики не дублируют данные, уже полноценно представленные в logs.
- Transformer не вычисляет Consumer-owned target statistics и не создаёт
  отдельную OpenSearch projection для metadata локального artifact.

## Границы инструментации

- OpenSearch-зависимый код остаётся в infrastructure/telemetry slice.
- Domain и application use cases не зависят от OpenSearch API;
  provider projection читается через capability-oriented port.
- Инструментируются существующие execution boundaries; прикладной код не
  перестраивается только ради observability.
- Сбор не распространяется на несвязанные компоненты без согласования scope.
- Core progress, необходимый для lifecycle или recovery, не становится
  telemetry только из-за сходства с метрикой.

## Надёжность

- Сбор и доставка имеют bounded потребление памяти, disk и времени.
- Очереди, retry budget и retention конечны; недоступность OpenSearch не
  создаёт неограниченную задержку основной операции.
- Повреждённое или отброшенное observation допустимо и не изменяет результат
  fit, публикацию модели или lifecycle job.
- Неполная, отсутствующая или повреждённая projection меняет только
  outcome Training Telemetry Query и не инвалидирует модель.
- Некорректная конфигурация publisher-а отключает только доставку и отражается
  в operational logs/counters.
- Централизованная projection ограничена определёнными metrics contracts epoch
  points и terminal run summary.

## Согласованность между проектами

Если разные компоненты экосистемы измеряют одну операцию или понятие, их имена,
единицы, семантика полей и структура документов должны быть совместимы.
Общий metrics contract нельзя изменять только в одном проекте, если это
создаёт расхождение эквивалентных метрик между компонентами.
