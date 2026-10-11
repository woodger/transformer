# Transformer

Transformer обучает модели PyTorch и выполняет предсказания на отдельном
сервере. Клиент передаёт данные и параметры через Apache Arrow Flight, получает
идентификатор задания и отслеживает его результат.

Сервис сохраняет задания в PostgreSQL, хранит обученные модели и контрольные
точки, управляет CPU и GPU и собирает показатели обучения. В связке с Inventory
расчёт признаков и смысл целевых значений остаются в Inventory, а обучение и
исполнение моделей — в Transformer.

## Возможности

- Обучение и предсказания на CPU или GPU. Вычисления идут в отдельном процессе
  Worker, чтобы сбой модели не завершал сетевой сервис.
- Настройка целей и функции потерь через `ModelContract`, включая взвешенную
  бинарную классификацию и регуляризацию энтропией.
- Компактная передача данных Arrow в формате `indexedFeatureBlocks` с
  восстановлением исходного порядка строк и признаков.
- Каталог обученных моделей, схема модели, показатели обучения и диагностика
  выходных головок. Эти запросы доступны только для чтения и учитывают владельца.
- Аутентификация по Bearer-токенам.

Обучение и предсказания запускаются через Flight. Локальные команды служат для
запуска сервиса и администрирования.

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

`flight serve` запускает сервис. Остальные команды управляют токенами доступа,
обученными моделями и миграциями PostgreSQL. Точные параметры показывает
`transformer <command> --help`.

## Документация

- [Начало работы](./docs/getting-started.md)
- [Интеграция с Flight](./docs/flight-integration.md)
- [Эксплуатация и администрирование](./docs/operations/index.md)
- [Развёртывание через systemd](./docs/deployment/systemd.md)
- [Функции потерь и регуляризация](./docs/losses.md)
- [Обучение и восстановление](./docs/training-runtime.md)
- [Аутентификация](./docs/authentication.md)
- [Архитектура](./docs/architecture.md)
- [Архитектурные решения](./docs/adr/index.md)

## API и совместимость

Сервис поддерживает протокол Flight v24. Предыдущие версии запросов,
определений моделей, каталога и телеметрии не поддерживаются. Точные форматы
документов и условия совместимости описаны в спецификациях:

- [Запросы Flight](./app/contracts/flight/v24/README.md)
- [Цели и функция потерь](./app/contracts/semantic/v7/README.md)
- [Каталог моделей](./app/contracts/model_catalog/v9/README.md)
- [Схема модели](./app/contracts/model_topology/v5/README.md)
- [Показатели обучения](./app/contracts/training_telemetry/v4/README.md)
- [Диагностика выходных головок](./app/contracts/target_head_diagnostics/v5/README.md)

Версии внутреннего протокола Worker, контрольных точек и хранения метрик
перечислены в [описании архитектуры](./docs/architecture.md).
