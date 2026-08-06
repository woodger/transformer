# Политика скриптов и точек запуска

> Type: Policy. Этот документ защищает CLI entrypoint, runtime wiring,
> subprocess protocol и операторские команды.

Проект не имеет отдельного build step. Основная точка запуска:

```text
.venv/bin/python app/main.py <command>
```

`app/main.py` выполняет dispatch и лениво подключает command-specific
dependencies. CLI schema и help находятся в `app/cli/`, handlers — в
`app/commands/`, а process-specific roots — в `app/service/bootstrap/`,
`app/worker/bootstrap/` и `app/admin/bootstrap/`.

## Что считается контрактом

- путь и форма запуска команд;
- порядок разбора аргументов и инициализации ресурсов;
- разделение текстового и бинарного stdout/stderr;
- framed stdin/stdout protocol;
- worker subprocess arguments, `cwd`, environment и process-group lifecycle;
- Alembic commands и ручной порядок применения migrations;
- systemd unit, описанный в `docs/deployment/systemd.md`.

## Запрещено без прямой необходимости

- добавлять shell wrapper, Makefile или install script только для удобства;
- выполнять migrations или cleanup автоматически при старте сервиса;
- удалять `models/`, `recovery/`, `/tmp/transformer` или output-файлы перед
  тестом/запуском;
- менять `app/main.py` ради unrelated refactoring;
- объединять команды через shell или запускать subprocess с `shell=True`;
- писать diagnostics в бинарный stdout `predict-stream`;
- менять рабочую директорию или Python executable worker-а без проверки
  recovery и checkpoint paths;
- автоматизировать ручной production deployment.

Subprocess следует запускать списком аргументов, с явным lifecycle и
ограниченным набором входных параметров. Для дочернего Python предпочтителен
текущий interpreter или уже зафиксированный project path, а не случайный
`python` из `PATH`.

## Изменение точки запуска

Изменение допустимо, если этого требует пользовательский сценарий или
исправление контракта. Одновременно нужно обновить:

- CLI tests и command-specific help;
- README или профильный behavioral reference;
- internal worker contract и worker bootstrap, если изменён process invocation;
- systemd-документацию, если изменился service entrypoint.

Скрипты не должны использоваться для маскировки проблемы окружения или
production-кода.
