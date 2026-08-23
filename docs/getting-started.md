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

## Настроить аутентификацию Flight

Flight service не выдаёт и не хранит API credentials. Для запуска сервиса и
административных OAuth client commands задайте базовый адрес Ory Hydra Admin
API:

```dotenv
HYDRA_ENDPOINT=http://hp260g9.home:4445
```

Transformer сам использует фиксированный path
`/admin/oauth2/introspect`.

Создать OAuth client Consumer-а можно отдельной короткоживущей admin-командой:

```bash
./.venv/bin/python ./app/main.py auth clients create consumer \
  --name="Consumer Transformer"
./.venv/bin/python ./app/main.py auth clients list
```

`--name` необязателен и по умолчанию равен `CLIENT_ID`. Успешный `create`
печатает точные client credentials один раз. Сохраните их
непосредственно в secret store Consumer-а; Transformer их не сохраняет и
Flight service их не получает. Consumer самостоятельно запрашивает
short-lived opaque access tokens через OAuth `client_credentials` с явными
`audience=transformer` и `scope=transformer:invoke`. Удаление client-а и отзыв
всех его tokens выполняются командой `auth clients delete consumer`. Error
semantics и полный порядок операций описаны в
[Flight runbook](./flight-operations.md#ory-hydra).

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
- [Arrow Flight v5 contract](../app/contracts/flight/v5/README.md) задаёт
  public remote API; [Flight runbook](./flight-operations.md) описывает
  PostgreSQL, OAuth, recovery, TLS и lifecycle service.
- [systemd guide](./deployment/systemd.md) — единственный ручной production
  deployment path для Fedora.
- [OpenSearch guide](./deployment/opensearch.md) — необязательная доставка
  run-owned training metrics после durable publication.

## Проверка изменений

Основной набор тестов самодостаточен; GPU-сценарии запускаются отдельно. Полный
набор штатных команд находится в единственном нормативном источнике —
[политике тестирования](./policy/testing-policy.md#запуск).
