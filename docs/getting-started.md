# Начало работы

> Type: Reference. Локальный сценарий для working copy Transformer Arrow Flight
> service.

Этот документ описывает запуск локального CLI. Для remote Arrow Flight service
с PostgreSQL используйте [runbook](./flight-operations.md), для production
systemd — [deployment guide](./deployment/systemd.md).

## Подготовить окружение

Из корня проекта создайте чистое virtual environment от системного
`/usr/bin/python3` и установите зафиксированные зависимости:

```bash
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
```

Владелец development-среды отвечает за подходящую версию системного Python.
Проект использует только `.venv`: application dependencies не устанавливаются
в system Python или user-site. Полные правила находятся в
[политике Python runtime](./policy/python-runtime-policy.md).

При изменении `requirements.txt` или системного `/usr/bin/python3` повторите
этот сценарий. Не используйте для приложения bare `pip` или `pip3`.

## Проверить CLI

```bash
./.venv/bin/python ./app/main.py --help
./.venv/bin/python ./app/main.py --version
```

Точные arguments, defaults и примеры leaf-команд показывает
`./.venv/bin/python ./app/main.py <command> --help`. Краткая карта команд и их
поведение собраны в [справочнике CLI](./cli/index.md).

## Локальное обучение и prediction

Входной файл — самостоятельный Arrow IPC file с колонками, описанными в
[локальном Arrow contract](./local-arrow-protocol.md). Пример обучения:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --device=cpu \
  --checkpoint-out=model_weights.pth \
  --seq-len=20 \
  --mode=relaxed \
  --epochs=25 \
  --batch-size=256
```

Prediction использует созданный checkpoint:

```bash
./.venv/bin/python ./app/main.py predict ./data/test.arrow \
  --device=cpu \
  --checkpoint=model_weights.pth \
  --output=/tmp/preds.arrow \
  --pred-col=out
```

Относительные checkpoint и metrics paths принадлежат project `models/`;
детали записи артефактов, совместимости checkpoint и training semantics — в
[справочнике CLI](./cli/index.md) и
[training reference](./training-runtime.md).

## Следующие сценарии

- [`fit-stream` и `predict-stream`](./local-arrow-protocol.md) принимают и
  возвращают framed Arrow payloads через standard streams.
- [Arrow Flight v2 contract](../app/contracts/flight/v2/README.md) задаёт
  public remote API; [Flight runbook](./flight-operations.md) описывает
  PostgreSQL, tokens, recovery, TLS и lifecycle service.
- [systemd guide](./deployment/systemd.md) — единственный ручной production
  deployment path для Fedora.

## Проверка изменений

PostgreSQL integration tests требуют отдельную базу, имя которой начинается с
`transformer_test`. Pytest создаёт в ней одноразовую schema и удаляет её после
session; production database намеренно отклоняется.

```bash
./.venv/bin/python -m ruff check .
POSTGRES_DB=transformer_test ./.venv/bin/python -m pytest -q
```

