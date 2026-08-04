# Политика тестирования

> Type: Policy. Этот документ задаёт требования к pytest tests и test
> infrastructure Transformer.

Тест защищает наблюдаемое поведение, persisted/wire contract или существенный
production-риск. Он не должен фиксировать случайную структуру реализации.

Основной вопрос:

> Какой дефект или контракт обнаружит этот тест?

## Когда тест обязателен

Тест нужен для изменения:

- numerical/model behavior, masking, losses и scheduler;
- Arrow IO, shape/dtype validation и framed protocol;
- CLI parsing, help, stdout/stderr и exit behavior;
- checkpoint serialization и backward compatibility;
- filesystem atomicity и path safety;
- Flight actions, authentication, idempotency и quotas;
- PostgreSQL state transitions, migrations и token cache;
- subprocess, cancellation, recovery и shutdown;
- configuration и environment parsing.

Тривиальный passthrough без собственного branching или mapping может не иметь
отдельного теста, если он покрыт через public scenario.

## Уровни тестов

### Unit tests

Проверяют pure/reusable behavior с минимальными данными: validators, losses,
config parsing, state transitions, serialization helpers.

### Integration tests

Проверяют реальную границу, когда она является сутью поведения:

- Arrow files и framed streams;
- filesystem, atomic rename и symlinks;
- loopback Flight server и TLS/mTLS;
- PostgreSQL transactions и Alembic;
- worker subprocess и process groups.

Интеграционный тест должен явно ограничивать resources и гарантированно
завершать threads/processes в teardown.

### Contract tests

Проверяют normative schemas и golden fixtures в `contracts/flight/v2/`.
Fixture обновляется только при намеренном изменении contract, а не ради
«починки» падающего теста.

JSON Schemas проверяются как Draft 2020-12 через `jsonschema`; локальные
`$ref` разрешаются только из `contracts/flight/v2/schemas/`.

## Структура и именование

Используется pytest:

- файл — `tests/test_<subject>.py`;
- функция — `test_<expected_behavior>`;
- fixture — существительное, описывающее предоставляемый resource;
- `@pytest.mark.parametrize` — для одной семантики на наборе inputs.

Хорошо:

```python
def test_disabled_plaintext_requires_tls(tmp_path):
    with pytest.raises(ValueError, match="plaintext Flight is disabled"):
        FlightServiceConfig(
            runtime_dir=str(tmp_path),
            allow_plaintext=False,
        ).validate()
```

Плохо:

```python
def test_config_works():
    assert build_config() is not None
```

Имя не обязано повторять class/function under test, если module уже задаёт
контекст. Оно должно прямо описывать результат или отказ.

Один тест проверяет одно поведение. Несколько assertions допустимы, когда вместе
описывают один contract result.

## Тестовые данные

Данные должны быть минимальными, но реалистичными для проверяемой границы.

- не создавать большую Arrow table, если достаточно двух rows;
- не строить полную модель, если проверяется parser;
- использовать маленькие tensors и фиксированный seed;
- явно включать NaN/Infinity/empty input только когда это часть сценария;
- давать fixture имена, раскрывающие нестандартные значения.

Проверка «не упало» недостаточна. Нужен точный state, output, exception,
serialized field или side effect.

## Изоляция и детерминизм

Тест не зависит от:

- порядка запуска;
- wall-clock timing без bounded synchronization;
- production credentials и внешней сети;
- общего mutable filesystem path;
- случайного GPU;
- artifacts предыдущего запуска.

Используются `tmp_path`, `monkeypatch`, `capsys` и явные fixtures. Время,
случайность и device availability контролируются через injection или
минимальный boundary mock.

Для floating point используются обоснованные tolerances. Exact equality
применяется к integer/state/wire contracts, а не к нестабильному численному
результату.

## PostgreSQL

PostgreSQL tests используют реальный PostgreSQL и отдельную disposable schema
вида `transformer_test_<uuid>`. Schema создаётся migrations и удаляется после
test session.

Запрещено:

- направлять тесты на production database;
- заменять PostgreSQL поведение SQLite;
- переиспользовать production schema `transformer`;
- оставлять test schema после успешного teardown;
- мокать transaction semantics в тесте, который заявлен как database
  integration.

Параметры подключения читаются тем же механизмом, что и приложение. Секреты не
попадают в fixtures, assertions или logs.

## Filesystem и artifacts

- временные файлы создаются только внутри `tmp_path`;
- published model, recovery store и runtime spool в тесте используют разные
  roots;
- path traversal и symlink escape проверяются явно;
- atomicity tests моделируют failure до и после replace/fsync;
- тест не читает и не изменяет project `models/`, `recovery/` или
  `/tmp/transformer`.

Cleanup не должен скрывать partial artifact, который и является предметом
assertion.

## Subprocess и concurrency

Тест process lifecycle обязан:

- использовать bounded timeout;
- завершать всю process group, если она создана;
- дожидаться reap дочернего процесса;
- не оставлять background thread;
- предпочитать event/condition polling фиксированному `sleep`;
- проверять state transition вместе с OS-level effect.

Для Flight cancellation и shutdown важно тестировать race boundaries, а не
только факт вызова mock.

## Что особенно важно для Transformer

### Training и model

- shape и feature dimension;
- all-masked и partially masked sequences;
- loss stages и global optimizer step;
- early stopping и best checkpoint;
- job-wide epochs поверх всех payloads;
- deterministic mode и explicit device policy.

### Arrow и stream

- empty table/frame;
- clean EOF и zero terminator;
- big-endian length и frame size limit;
- сохранение logical payload boundaries;
- binary stdout без diagnostics;
- schema, dtype, width, NaN и Infinity.

### Flight service

- missing/invalid authentication;
- duplicate, out-of-order и repeated requests;
- lost-response replay и idempotency conflict;
- quota и фактические `ENOSPC`/`EDQUOT` storage failures;
- valid/invalid job state transitions;
- cancellation до и во время worker execution;
- ordinary restart и runtime storage epoch loss;
- resume с completed-global-epoch, missing/corrupt recovery artifacts и
  cancellation races вокруг retry;
- physical GPU assignment, confirmed device loss, boot-scoped quarantine и
  reassignment на другой healthy GPU;
- atomic model publication и opaque `modelRef`;
- token cache refresh без database query на каждый RPC.

Нужны только сценарии, соответствующие изменяемой ответственности; checklist не
означает дублирование всех cases в каждом module.

## Mocks и fakes

Mock используется для дорогой или внешней boundary, но не должен повторять
алгоритм production-кода.

Предпочтительно:

- fake server с наблюдаемым lifecycle;
- injected clock/device probe;
- monkeypatch конкретного filesystem failure;
- small in-memory stream для framed protocol.

Нежелательно:

- проверять только количество вызовов, когда важен результат;
- мокать ORM в тесте транзакционного поведения;
- копировать validation algorithm в expected calculation;
- тестировать private helper напрямую, если доступен public path.

## Regression tests

Regression test получает имя по защищаемому поведению:

```python
def test_fit_stream_runs_epochs_over_all_payloads():
    ...
```

Имя вроде `test_bug_from_last_release` не фиксирует контракт. История дефекта
остаётся в commit/MR, а тест — в текущем ожидаемом поведении.

## Запуск

Сначала запускается изменённый module или группа:

```bash
.venv/bin/python -m pytest -q tests/test_flight_config.py
```

Перед release и после изменений общих contracts:

```bash
.venv/bin/python -m pytest -q
```

Skipped test не считается доказательством проверенного поведения. Причина skip
должна быть конкретной: например, отсутствие `openssl` для TLS integration.
Результат проверки сообщается как passed, failed, not run или blocked by
environment.
