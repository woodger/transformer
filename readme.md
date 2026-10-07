# Сервис Transformer Arrow Flight

Transformer — Python-сервис для надёжного обучения и предсказания с помощью
PyTorch через аутентифицированные задания Apache Arrow Flight. Inventory —
ориентированное на браузер приложение; Transformer владеет исполнением модели,
хранением, checkpoint-ами, восстановлением и проекцией телеметрии.

## Актуальная граница

- Flight v23 — единственный публичный workflow заданий.
- Semantic v6 содержит упорядоченные непрозрачные targets, явные bindings
  objective и настройку модели без внешних предметных литералов
  архитектуры. Для строго бинарной цели прямого компонента доступен отдельный
  примитив `PositiveClassWeightedBinaryCrossEntropyWithLogits` с явным весом
  положительного класса и автоматической проекцией вероятности исходного
  распределения.
- Model Catalog Query v8, Model Topology Query v4, Training Telemetry Query
  v4 и Target Head Diagnostics Query v5 — owner-scoped поверхности только для
  чтения, активируемые Flight v23.
- Worker v21, checkpoint/recovery v13 и metrics v12 являются внутренними для
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

- [Flight v23](./app/contracts/flight/v23/README.md)
- [Семантическая модель v6](./app/contracts/semantic/v6/README.md)
- [Запрос каталога моделей v8](./app/contracts/model_catalog/v8/README.md)
- [Запрос topology модели v4](./app/contracts/model_topology/v4/README.md)
- [Запрос телеметрии обучения v4](./app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок v5](./app/contracts/target_head_diagnostics/v5/README.md)
- [Worker v21](./app/contracts/worker/v21/README.md)
- [Checkpoint/recovery v13](./app/contracts/checkpoint/v13/README.md)
- [Архитектура](./docs/architecture.md)
- [Интеграция с Flight](./docs/flight-integration.md)
- [Эксплуатация Flight](./docs/operations/flight-service.md)
- [Развёртывание](./docs/deployment/systemd.md)
- [Архитектурные решения](./docs/adr/index.md)
