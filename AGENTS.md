# Правила работы с Transformer

Этот файл маршрутизирует агента к нормативным источникам проекта. Поведение,
contracts и архитектурные причины не дублируются здесь.

## Карта проекта

- `app/service/` — Flight service: domain, application, adapters и bootstrap;
- `app/worker/` — Arrow/Torch data path, model, training и checkpoints;
- `app/contracts/flight/v4/` — текущий публичный Flight contract;
- `app/contracts/worker/v3/` — текущий внутренний process contract;
- `app/admin/` — access tokens и database commands;
- `docs/policy/` — правила изменения production-кода.

## Что прочитать перед изменением

- model, tensors, training или masks — `docs/policy/typing-policy.md`,
  `docs/training-runtime.md` и `docs/losses.md`;
- local Arrow/file/stream — `docs/local-arrow-protocol.md`;
- Flight — `app/contracts/flight/v4/README.md`;
- worker process — `app/contracts/worker/v3/README.md`;
- service boundary — `docs/policy/architecture.md`;
- CLI/runtime wiring — `docs/policy/scripts-policy.md`.

## Проверка

Для обычного изменения Python:

```bash
./.venv/bin/python -m ruff check .
./.venv/bin/pyright
./.venv/bin/python -m pytest -q <затронутые tests>
```

Быстрый CPU-набор без внешних ресурсов:

```bash
./.venv/bin/python -m pytest -q -m "not gpu and not postgres"
```

PostgreSQL tests запускаются только с выделенной БД, имя которой начинается с
`transformer_test`. GPU tests запускаются только при изменении CUDA, AMP,
model execution или GPU runtime и требуют реальное CUDA-устройство.

После изменения versioned contract, общей training semantics или перед
release запускается полный доступный suite. Skip, вызванный отсутствием
реального GPU, указывается в результате проверки.

## Готовность

Изменение готово, когда сохранены публичные contracts вне заявленного scope,
обновлены непосредственно затронутые docs/tests, Ruff и Pyright проходят, а
результаты pytest сообщены без сокрытия skipped или environment blockers.
