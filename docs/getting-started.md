# Начало работы

> Тип: руководство. Подготовка working copy и запуск Transformer Arrow Flight
> service.

Transformer не предоставляет local fit/predict CLI. Обучение и prediction
начинаются только через authenticated Flight v15 Consumer boundary.

## Подготовить окружение

```bash
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python app/main.py --help
```

Используйте только project `.venv`. Версию системного Python и системные
зависимости выбирает владелец environment; application dependencies не
устанавливаются в system Python или user-site.

## Подготовить control plane

Настройте PostgreSQL и environment согласно
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

Для production используйте systemd procedure. Перед первым Flight v15 release
изучите destructive clean-cut instructions для migration 0027: старые jobs,
models, recovery records, и telemetry не сохраняются.

Consumer materializes a Semantic v3 `ModelContract`, creates a v15 fit or
predict job, uploads compact `indexedFeatureBlocks`, closes input, and polls
the issued job. Contract shape is documented in
[Flight v15](../app/contracts/flight/v15/README.md); this guide deliberately
does not duplicate wire examples.

## Проверка изменений

Точные validation commands находятся в
[политике тестирования](./policy/testing-policy.md#запуск). GPU-specific
verification выполняют на GPU deployment host, когда задача затрагивает CUDA
runtime.
