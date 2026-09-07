# Owner-scoped Training Telemetry Query

> Тип: Design Note. Consumer-side и Transformer-side граница read-only
> training telemetry согласована. Точный staged contract опубликован в
> [`app/contracts/training_telemetry/v1`](../../app/contracts/training_telemetry/v1/README.md).

- Статус: canonical package v1 опубликован для точного Consumer review
- Срез: 2026-09-07, действующий runtime Flight v12
- Область: будущий query contract; runtime, migrations и deployment не изменены

## Согласованный результат

Inventory запрашивает telemetry по exact `modelRef`, а Transformer выводит
owner и producing run из registry. Query возвращает только полный проверенный
report успешного fit либо явные `pending`/`unavailable` outcomes. Он не
превращает telemetry в model truth и не раскрывает OpenSearch topology.

Epoch values являются метриками training-прохода: каждая партия наблюдается до
своего optimizer update. Повторной оценки published checkpoint нет. Gradient
interactions загружаются отдельным bounded запросом одной epoch.

Model Catalog, fit/predict, metrics v5 и checkpoint integrity остаются
отдельными границами. Package резервирует action identities, но до его принятия
со стороны Inventory hosting Flight version, persistence и runtime
implementation не определяются.
