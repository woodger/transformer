# Начало работы

> Тип: руководство. Подготовка working copy и запуск Transformer Arrow Flight
> service.

Transformer не предоставляет локальный CLI fit/predict. Обучение и prediction
начинаются только через аутентифицированную границу Consumer Flight v15.

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
[операционному руководству](./operations/flight-service.md), затем проверьте и
примените migrations:

```bash
./.venv/bin/python app/main.py db migrations status
./.venv/bin/python app/main.py db migrations apply
```

Выпустите bearer credential для service identity Consumer-а:

```bash
./.venv/bin/python app/main.py auth tokens issue
```

Credential выводится один раз. Храните его в secret storage, не в repository,
command history, или логах.

## Запустить service

```bash
./.venv/bin/python app/main.py flight serve --host=127.0.0.1 --port=8815
```

Для production используйте процедуру systemd. Перед первым релизом Flight v15
изучите инструкции destructive clean cut для migration 0027: старые jobs,
models, recovery records и telemetry не сохраняются.

Consumer материализует `ModelContract` Semantic v3, создаёт job fit или
predict v15, загружает compact `indexedFeatureBlocks`, закрывает input и
опрашивает выпущенное job. Форма contract описана в
[Flight v15](../app/contracts/flight/v15/README.md); это руководство намеренно
не дублирует wire examples.

## Проверка изменений

Точные команды validation находятся в
[политике тестирования](./policy/testing-policy.md#запуск). GPU-specific
verification выполняют на deployment host с GPU, когда задача затрагивает
runtime CUDA.
