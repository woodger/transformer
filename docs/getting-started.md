# Начало работы

> Тип: руководство. Подготовка working copy и запуск Transformer Arrow Flight
> service.

Transformer не предоставляет локальный CLI fit/predict. Обучение и prediction
начинаются только через аутентифицированную границу вызывающей системы Flight
v24.

## Подготовить окружение

```bash
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python app/main.py --help
```

Используйте только `.venv` проекта. Версию системного Python и системные
зависимости выбирает владелец окружения; зависимости приложения не
устанавливаются в системный Python или пользовательский site.

## Подготовить control plane

Настройте PostgreSQL и окружение согласно
[операционному руководству](./operations/flight-service.md#конфигурация-окружения),
затем проверьте и примените migrations:

```bash
./.venv/bin/python app/main.py db migrations status
./.venv/bin/python app/main.py db migrations apply
```

Выпустите bearer credential для вызывающей service identity:

```bash
./.venv/bin/python app/main.py auth tokens issue
```

Credential выводится один раз. Храните его в secret storage, не в repository,
command history, или логах.

## Запустить service

```bash
./.venv/bin/python app/main.py flight serve --host=127.0.0.1 --port=8815
```

Для production используйте процедуру systemd. Migration 0031 является
destructive clean cut для перехода на Semantic v7. Перед запуском Flight v24
примените процедуру `docs/operations/database-migrations.md`. Для этого
перехода также замените индексы metrics v12 на v13 по
`docs/deployment/opensearch.md`.

Вызывающая система материализует `ModelContract` Semantic v7, создаёт job fit
или predict v24, загружает compact `indexedFeatureBlocks`, закрывает input и
опрашивает выпущенное job. Форма contract описана в
[Flight v24](../app/contracts/flight/v24/README.md); это руководство намеренно
не дублирует wire examples.

## Проверка изменений

Точные команды validation находятся в
[политике тестирования](./policy/testing-policy.md#запуск). GPU-specific
verification выполняют на deployment host с GPU, когда задача затрагивает
runtime CUDA.
