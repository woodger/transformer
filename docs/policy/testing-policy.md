# Политика тестирования

> Type: Policy. Этот документ задаёт текущую границу pytest tests Transformer.

Тест защищает наблюдаемое поведение, persisted/wire contract или существенный
production-риск. Он не должен фиксировать случайную структуру реализации.

Основной вопрос:

> Какой дефект или контракт обнаружит этот тест?

## Текущая граница

Основной набор полностью самодостаточен. Он не требует PostgreSQL, OpenSearch,
внешней сети, credentials или заранее подготовленных системных ресурсов и не
содержит условно пропускаемых тестов. Внутри процесса допустимы временные файлы,
loopback Flight server и bounded child processes: тест сам создаёт их и всегда
освобождает.

GPU-сценарии выделены единственным resource marker `gpu` и запускаются явно.
Обычный запуск исключает их через конфигурацию `pyproject.toml`.

Полноценные PostgreSQL и сквозные integration tests будут спроектированы после
появления выделенной инфраструктуры. Сейчас в проекте нет `postgres` marker,
`TEST_POSTGRES_URL`, скрытого PostgreSQL profile или тестов, которые молча
пропускаются из-за недоступной database. Отсутствующую инфраструктуру не
подменяют SQLite или ORM mocks, претендующие на проверку transaction semantics.

Отдельные Alembic revisions и migration chain не входят в pytest boundary.
Нельзя добавлять tests, которые импортируют revision modules, подменяют
`alembic.op`, сравнивают migration metadata или SQL либо фиксируют номер
текущего head. Административные команды migrations можно тестировать на уровне
dispatch и presentation без проверки DDL. Корректность schema migration
проверяется применением на выделенном PostgreSQL; отсутствие такой
infrastructure фиксируется как непроверенное ограничение и не компенсируется
unit test или mock database.

## Запуск

Основной набор:

```bash
./.venv/bin/python -m pytest
```

GPU-сценарии:

```bash
./.venv/bin/python -m pytest -m gpu
```

Это полный набор штатных pytest-команд проекта. Дополнительные marker profiles
и специальные команды для отдельных видов integration tests не поддерживаются.

GPU test может быть фактически проверен только в среде с доступным CUDA device.
Skip в такой среде не считается успешной проверкой GPU-поведения.

## Статическая проверка

Ruff проверяет style, imports и выбранные defect patterns. Pyright проверяет
типизированный scope, зафиксированный в `pyproject.toml`. Правила типов и tensor
runtime contracts находятся в [политике типов](./typing-policy.md).

Статические проверки дополняют pytest, но не образуют дополнительные test
profiles. Pyright не заменяет runtime tests shape, dtype, NaN/Infinity,
CUDA/AMP и serialization.

## Структура

Тесты группируются по проверяемой границе и владельцу:

```text
tests/unit/{admin,cli,local,service,worker}
tests/contract/{flight_v5,worker_v7,metrics_v3,metrics_fit_run_v2}
tests/integration/{flight,worker_process}
tests/architecture
tests/support
```

Каталог `tests/integration` содержит только самодостаточные process-boundary
сценарии, которые входят в основной быстрый набор. Он не означает наличие
внешней integration infrastructure.

Правила именования:

- файл — `tests/<level>/<owner>/test_<subject>.py`;
- функция — `test_<expected_behavior>`;
- fixture — существительное, описывающее предоставляемый resource;
- `@pytest.mark.parametrize` — для одной семантики на наборе inputs.

Один тест проверяет одно поведение. Несколько assertions допустимы, когда вместе
описывают один contract result. Проверка «не упало» недостаточна: нужен точный
state, output, exception, serialized field или side effect.

Предпочтительны многочисленные небольшие семантические тесты. Тяжёлый runtime,
subprocess или полный process boundary не должны повторяться для каждого такого
теста: один неизменяемый диагностический снимок может обслуживать несколько
независимых assertions. Настоящий subprocess сохраняется там, где именно запуск,
изоляция или lifecycle отдельного процесса являются проверяемым поведением.

Тест не должен зависеть от истории миграций проекта. Структурные тесты проверяют
актуальные позитивные инварианты: наличие canonical locations, направление
зависимостей и действующие process boundaries. Они не содержат отрицательных
assertions по прежним путям, именам packages и legacy source/contract-файлам.
Удаление таких artifacts проверяется при выполнении самого изменения и review,
но не закрепляется постоянным regression test.

Это ограничение относится именно к истории размещения кода. Оно не запрещает
отрицательные assertions для текущего наблюдаемого поведения, security boundary
или нормативного runtime-контракта.

## Тестовые данные и изоляция

Данные должны быть минимальными, но реалистичными для проверяемой границы.

- использовать маленькие tensors, Arrow tables и фиксированный seed;
- явно включать NaN, Infinity и empty input только как часть сценария;
- использовать `tmp_path`, `monkeypatch` и явные fixtures;
- не зависеть от порядка запуска и artifacts предыдущего запуска;
- не читать production credentials и не обращаться во внешнюю сеть;
- не использовать wall-clock ожидание без bounded synchronization.

Для floating point применяются обоснованные tolerances. Exact equality
используется для integer, state и wire contracts, а не для нестабильного
численного результата.

## Contracts

Contract tests проверяют нормативные schemas и golden fixtures Flight v5,
worker v7 и metrics contracts. Fixture обновляется только при намеренном
изменении contract, а не ради прохождения падающего теста.

Cross-language проверки используют локальный Node.js без npm dependencies и
сравнивают RFC 8785/JCS digest с Python runtime. JSON Schemas проверяются как
Draft 2020-12 через `jsonschema`; runtime ingress использует те же schemas.

## Filesystem, subprocess и concurrency

- временные файлы создаются только внутри `tmp_path`;
- atomicity tests моделируют failure до и после replace/fsync;
- process lifecycle имеет bounded timeout и reap дочернего процесса;
- созданная process group завершается целиком;
- background threads закрываются в teardown;
- event/condition polling предпочтительнее фиксированного `sleep`.

Тест не должен оставлять процессы, threads, sockets или файлы после завершения.

## Mocks и regression tests

Mock применяется для узкой внешней boundary, но не повторяет production
алгоритм. Транзакционное поведение PostgreSQL нельзя считать проверенным через
ORM mock.

Regression test получает имя по защищаемому поведению, например:

```python
def test_fit_stream_runs_epochs_over_all_payloads():
    ...
```

История дефекта остаётся в commit/MR, а имя теста описывает текущее ожидаемое
поведение.

## Достоверность результата

Проверка считается пройденной только после фактически успешного запуска.
Результат сообщается как `passed`, `failed`, `not run` или `blocked by
environment`. Skipped test не является доказательством проверенного поведения.
