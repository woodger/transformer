# Политика Python runtime и виртуальных окружений

> Type: Policy. Этот документ задаёт ownership base interpreter, project
> virtual environment и Python package installation.

## Граница ответственности

Проект не выбирает minor-версию Python. Владелец development, CI или production
среды обеспечивает подходящий `/usr/bin/python3` и подтверждает совместимость
зависимостей с ним. Project runtime всегда находится в `.venv`, созданном этим
interpreter.

`requirements.txt` фиксирует версии Python-пакетов проекта, но не заменяет
ответственность владельца среды за system Python.

## Обязательные правила

- `/usr/bin/python3` — единственный base interpreter для создания project
  `.venv`. Project documentation, CLI help и scripts не выбирают versioned
  interpreter path.
- Каждая working copy и каждый production runtime используют собственный
  `.venv`. Его создают на target host и не копируют, не переносят и не
  коммитят в Git.
- Все application commands, tests и package operations используют явный
  project interpreter: `.venv/bin/python ...` и
  `.venv/bin/python -m pip ...`.
- Shell activation допустима только как интерактивное удобство. Scripts, CI и
  systemd не должны зависеть от `PATH` или activation state.
- Зависимости приложения нельзя устанавливать в system Python или user-site.
  В частности, запрещены `sudo pip`, bare `pip`/`pip3`, `pip install --user`
  и `--break-system-packages` для application dependencies.

## Создание и обновление окружения

Из корня project:

```bash
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
```

Если владелец среды изменяет system `/usr/bin/python3`, существующее `.venv`
нужно удалить и создать заново перед установкой lock-файла. In-place смена
base interpreter не допускается.

## Проверка

Перед запуском или deployment нужно подтвердить, что project interpreter
изолирован и зависимости согласованы:

```bash
./.venv/bin/python --version
./.venv/bin/python -m pip check
./.venv/bin/python -m pytest -q
```

Подробный production-порядок находится в
[`docs/deployment/systemd.md`](../deployment/systemd.md). Правила запуска
scripts и subprocess дополняет [Политика скриптов и точек запуска](./scripts-policy.md).
