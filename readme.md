# Сервис Transformer Arrow Flight

Transformer — Python-сервис для надёжного обучения и предсказания с помощью
PyTorch через аутентифицированные задания Apache Arrow Flight. Inventory —
ориентированное на браузер приложение; Transformer владеет исполнением модели,
хранением, checkpoint-ами, восстановлением и проекцией телеметрии.

## Актуальная граница

- Flight v24 — единственный публичный workflow заданий.
- Semantic v7 содержит упорядоченные непрозрачные targets, явные bindings
  objective и настройку модели без внешних предметных литералов
  архитектуры. Для строго бинарной цели прямого компонента доступен отдельный
  примитив `PositiveClassWeightedBinaryCrossEntropyWithLogits` с явным весом
  положительного класса и автоматической проекцией вероятности исходного
  распределения. Optional `BernoulliEntropyPenalty` минимизирует положительную
  энтропию публичной вероятности; `BernoulliConfidencePenalty` сохраняет
  отрицательную энтропию.
- Model Catalog Query v9, Model Topology Query v5, Training Telemetry Query
  v4 и Target Head Diagnostics Query v5 — owner-scoped поверхности только для
  чтения, активируемые Flight v24.
- Worker v22, checkpoint/recovery v14 и metrics v13 являются внутренними для
  provider-а.
- `indexedFeatureBlocks` остаётся компактным Arrow-представлением входных
  данных; его семантика логического восстановления не изменилась.

Локальный CLI для fit/predict и compatibility reader для прежних ревизий
Flight, semantic, model catalog или telemetry отсутствуют.

## Установка и проверка

Из корня репозитория:

```sh
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app/main.py --help
```

Интерпретатор проекта — `.venv/bin/python`. Не устанавливайте зависимости
приложения в системный Python или пользовательский site.

## Команды

```text
flight serve
auth tokens issue|list|revoke
models list|delete
db migrations status|apply|rollback
```

`flight serve` запускает удалённый сервис. Остальные команды — локальные
операционные команды для access tokens в базе данных, опубликованных
generations и состояния Alembic. Точные параметры показывает
`transformer <command> --help`.

## Документация

- [Flight v24](./app/contracts/flight/v24/README.md)
- [Семантическая модель v7](./app/contracts/semantic/v7/README.md)
- [Запрос каталога моделей v9](./app/contracts/model_catalog/v9/README.md)
- [Запрос topology модели v5](./app/contracts/model_topology/v5/README.md)
- [Запрос телеметрии обучения v4](./app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок v5](./app/contracts/target_head_diagnostics/v5/README.md)
- [Worker v22](./app/contracts/worker/v22/README.md)
- [Checkpoint/recovery v14](./app/contracts/checkpoint/v14/README.md)
- [Архитектура](./docs/architecture.md)
- [Интеграция с Flight](./docs/flight-integration.md)
- [Эксплуатация Flight](./docs/operations/flight-service.md)
- [Развёртывание](./docs/deployment/systemd.md)
- [Архитектурные решения](./docs/adr/index.md)
