# Начало работы

> Тип: руководство. Локальный сценарий для working copy Transformer Arrow
> Flight service.

Этот документ описывает запуск локального CLI. Для remote Arrow Flight service
с PostgreSQL используйте [runbook](./operations/flight-service.md), для production
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

## Создать API-токен

После настройки PostgreSQL и применения migrations выпустите bearer token для
клиентского service identity:

```bash
./.venv/bin/python ./app/main.py auth tokens issue
```

Команда один раз выводит token ID, expiration time и новый credential вида
`a.<base64url>`. Сохраните credential в secret storage клиентского приложения;
не помещайте его в repository, логи или server `.env`.

Просмотр, безопасную передачу Consumer-у, ротацию и отзыв описывает
[`руководство по управлению API access tokens`](./operations/api-access-tokens.md).
Persistence, cache и security semantics находятся в
[`справочнике аутентификации`](./authentication.md).

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
- [Arrow Flight v6 contract](../app/contracts/flight/v6/README.md) задаёт
  public remote API; [Flight runbook](./operations/flight-service.md) описывает
  PostgreSQL, recovery, TLS и lifecycle service.
- [Операционные руководства](./operations/index.md) описывают lifecycle API
  access tokens, PostgreSQL schema и published models.
- [systemd guide](./deployment/systemd.md) — единственный ручной production
  deployment path для Fedora.
- [OpenSearch guide](./deployment/opensearch.md) — необязательная доставка
  run-owned training metrics после durable publication.

## Проверка изменений

Основной набор тестов самодостаточен; GPU-сценарии запускаются отдельно. Полный
набор штатных команд находится в единственном нормативном источнике —
[политике тестирования](./policy/testing-policy.md#запуск).
