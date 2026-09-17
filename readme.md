# Сервис Transformer Arrow Flight

Transformer — Python-сервис для надёжного обучения и предсказания с помощью
PyTorch через аутентифицированные задания Apache Arrow Flight. Inventory —
ориентированное на браузер приложение; Transformer владеет исполнением модели,
хранением, checkpoint-ами, восстановлением и проекцией телеметрии.

## Актуальная граница

- Flight v16 — единственный публичный workflow заданий.
- Semantic v3 содержит упорядоченные непрозрачные targets, явные bindings
  objective и настройку модели без внешних предметных литералов
  архитектуры.
- Model Catalog Query v3, Model Topology Query v1 и Training Telemetry Query
  v3 — owner-scoped поверхности только для чтения, активируемые Flight v16.
- Worker v14, checkpoint/recovery v8 и metrics v7 являются внутренними для
  provider-а.
- `indexedFeatureBlocks` остаётся компактным Arrow-представлением входных
  данных; его семантика логического восстановления не изменилась.

Локальный CLI для fit/predict и compatibility reader для прежних ревизий
Flight, semantic, checkpoint, model catalog или telemetry отсутствуют.

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

- [Flight v16](./app/contracts/flight/v16/README.md)
- [Семантическая модель v3](./app/contracts/semantic/v3/README.md)
- [Запрос каталога моделей v3](./app/contracts/model_catalog/v3/README.md)
- [Запрос topology модели v1](./app/contracts/model_topology/v1/README.md)
- [Запрос телеметрии обучения v3](./app/contracts/training_telemetry/v3/README.md)
- [Worker v14](./app/contracts/worker/v14/README.md)
- [Checkpoint/recovery v8](./app/contracts/checkpoint/v8/README.md)
- [Архитектура](./docs/architecture.md)
- [Интеграция с Flight](./docs/flight-integration.md)
- [Эксплуатация Flight](./docs/operations/flight-service.md)
- [Развёртывание](./docs/deployment/systemd.md)
- [Архитектурные решения](./docs/adr/index.md)
