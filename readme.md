# Сервис Transformer Arrow Flight

Transformer — Python-сервис для надёжного обучения и предсказания с помощью
PyTorch через аутентифицированные задания Apache Arrow Flight. Inventory —
ориентированное на браузер приложение; Transformer владеет исполнением модели,
хранением, checkpoint-ами, восстановлением и проекцией телеметрии.

## Актуальная граница

- Flight v21 — единственный публичный workflow заданий.
- Semantic v4 содержит упорядоченные непрозрачные targets, явные bindings
  objective и настройку модели без внешних предметных литералов
  архитектуры. Для строго бинарной цели прямого компонента доступен отдельный
  примитив `PositiveClassWeightedBinaryCrossEntropyWithLogits` с явным весом
  положительного класса и автоматической проекцией вероятности исходного
  распределения.
- Model Catalog Query v6, Model Topology Query v2, Training Telemetry Query
  v4 и Target Head Diagnostics Query v3 — owner-scoped поверхности только для
  чтения, активируемые Flight v21.
- Worker v19, checkpoint/recovery v11 и metrics v10 являются внутренними для
  provider-а.
- `indexedFeatureBlocks` остаётся компактным Arrow-представлением входных
  данных; его семантика логического восстановления не изменилась.

Локальный CLI для fit/predict и compatibility reader для прежних ревизий
Flight, semantic, model catalog или telemetry отсутствуют. Model Catalog v6
имеет единственную узкую проекцию metadata checkpoint v9, чтобы такая model
получала `notConfigured` в Target Head Diagnostics v4.

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

- [Flight v21](./app/contracts/flight/v21/README.md)
- [Семантическая модель v4](./app/contracts/semantic/v4/README.md)
- [Запрос каталога моделей v6](./app/contracts/model_catalog/v6/README.md)
- [Запрос topology модели v2](./app/contracts/model_topology/v2/README.md)
- [Запрос телеметрии обучения v4](./app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок v4](./app/contracts/target_head_diagnostics/v4/README.md)
- [Worker v19](./app/contracts/worker/v19/README.md)
- [Checkpoint/recovery v11](./app/contracts/checkpoint/v11/README.md)
- [Архитектура](./docs/architecture.md)
- [Интеграция с Flight](./docs/flight-integration.md)
- [Эксплуатация Flight](./docs/operations/flight-service.md)
- [Развёртывание](./docs/deployment/systemd.md)
- [Архитектурные решения](./docs/adr/index.md)
