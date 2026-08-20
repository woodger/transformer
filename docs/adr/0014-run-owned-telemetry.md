# ADR 0014: run-owned training telemetry

- Статус: принято
- Дата: 2026-08-18
- Уточняет: ADR 0009, ADR 0010, ADR 0011, ADR 0012 и ADR 0013

## Контекст

После выделения telemetry slice сохранялись лишние связи с другими
владельцами ответственности:

- training target statistics вычислялись по запросу Consumer, хотя исходным
  dataset и baseline владеет Consumer;
- worker писал attempt-local `metrics.jsonl`, который service не читал;
- durable telemetry хранилась внутри каталога модели и ссылалась на неё
  внешними ключами PostgreSQL;
- удаление и список моделей знали о состоянии metrics outbox;
- OpenSearch получал отдельный artifact document без эксплуатационного
  читателя.

Эти связи не улучшали наблюдаемость, но позволяли telemetry проникать во
внутренний worker contract и lifecycle модели.

## Решение

Transformer сохраняет только собственные наблюдения обучения и выполнения:
loss, per-target errors, checkpoint selection, AMP/optimizer/gradient counters,
phase durations и terminal fit-run durations/counters. Статистики значений
training targets удаляются из worker v6 result и fit-run v2.

Service строит единственный durable набор успешного run:

```text
telemetry/<jobId>/metrics.jsonl
telemetry/<jobId>/run-summary.json
```

Каталог `telemetry/` является отдельным storage root рядом с `models/` и
`recovery/`. Он не вложен в каталог моделей и использует собственные проверки
relative path, reconciliation и retention.

PostgreSQL tables `training_metrics_artifacts`,
`fit_run_summary_artifacts` и `metrics_outbox` идентифицируются по `jobId`.
`modelRef` остаётся корреляционным полем без внешнего ключа к model lifecycle.
Миграция 0010 удаляет прежние model-owned artifact metadata и outbox: это
наблюдаемость экспериментального контракта, для которой compatibility surface
и перенос скрытых путей не сохраняются. Committed epoch intervals остаются в
PostgreSQL до штатной job retention.

Удаление модели работает только с checkpoint, model metadata, alias и строкой
модели. Оно не отменяет доставку и не удаляет run telemetry. Terminal outbox
retention удаляет telemetry metadata и точные локальные файлы; startup
reconciliation удаляет orphan run-каталоги.
Startup также удаляет устаревшие server-owned `metrics.jsonl` и
`run-summary.json` из каталогов моделей, включая промежуточный каталог
`models/_telemetry`.

OpenSearch projection содержит только `metrics-points-v2` и
`metrics-runs-v2`. Отдельный `metrics-artifacts-v2` удалён: локальная artifact
metadata остаётся внутренним механизмом integrity/retry и не является
аналитическим событием.

Worker больше не создаёт второй attempt-local metrics-файл. Локальные CLI
команды по-прежнему могут явно писать пользовательский `metrics.jsonl`.

## Инварианты

- Flight v4 и ML objective не меняются.
- Ошибка telemetry не меняет публикацию модели и terminal outcome fit.
- Удаление модели не влияет на состояние outbox.
- `jobId` является durable identity telemetry run; `modelRef` служит только
  корреляции.
- Transformer не вычисляет Consumer-owned dataset baselines.
- В OpenSearch не публикуются filesystem paths или отдельная metadata локальных
  artifacts.

## Последствия

Telemetry имеет собственные identity, storage, retention и delivery lifecycle.
Модель можно удалить до или после доставки метрик без скрытых side effects.
Consumer получает только согласованные observability documents, а не
вычисления над принадлежащими ему исходными данными.
