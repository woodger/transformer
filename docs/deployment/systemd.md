# Развёртывание через systemd

> Type: Reference. Первый production deployment `transformer flight serve` на
> Fedora через systemd.

Production tree находится только в `/opt/transformer` и запускается от
`nerv:nerv`. Рабочая копия `/home/nerv/transformer` — источник выбранного
release, но не runtime service. Такое разделение необходимо на Fedora с SELinux
в режиме `Enforcing`: systemd не должен исполнять файлы с типом `user_home_t`
из `/home`.

Команды ниже выполняются из interactive shell пользователя `nerv`; `sudo`
используется только там, где требуется изменить system-owned path или unit.

Инструкция описывает безопасный первый deployment. Она намеренно не
перезаписывает существующий `/opt/transformer`: обновление production tree,
моделей и recovery state требует отдельного согласованного change.

## Проверить исходную рабочую копию

Выберите commit, который нужно развернуть, в `/home/nerv/transformer`. Перед
копированием working copy должна быть чистой, а `requirements.txt` —
tracked-файлом выбранного commit:

```bash
cd /home/nerv/transformer
test "$(id -un)" = nerv
test -z "$(git status --porcelain)"
git diff --check
git ls-files --error-unmatch requirements.txt
git log -1 --oneline
test -f .env
```

Эти проверки не выбирают revision и не выполняют `git pull`: это ответственность
оператора. Они не дают случайно развернуть untracked lock-файл, временный unit
или незакоммиченный код.

## Создать production tree

Если unit уже был установлен или преждевременно включён, сначала остановите и
отключите его: иначе `Restart=on-failure` будет пытаться запускать неполное
дерево. Затем создайте новый root. Команда `test` намеренно прерывает сценарий,
если `/opt/transformer` уже существует.

```bash
sudo systemctl disable --now transformer 2>/dev/null || true
test ! -e /opt/transformer
sudo install -d -o nerv -g nerv -m 0755 /opt/transformer

git -C /home/nerv/transformer archive --format=tar HEAD | \
  tar -C /opt/transformer -xf -

install -m 0600 /home/nerv/transformer/.env /opt/transformer/.env

for runtime_directory in models recovery; do
  if test -d "/home/nerv/transformer/$runtime_directory"; then
    tar -C /home/nerv/transformer -cf - "$runtime_directory" | \
      tar -C /opt/transformer -xf -
  fi
done

install -d -m 0700 /opt/transformer/models /opt/transformer/recovery
```

`git archive` переносит только tracked files выбранного commit. `.env`,
`models/` и `recovery/` намеренно копируются отдельно: они игнорируются Git,
но принадлежат runtime. `.venv`, caches, pending unit files и untracked
`requirements.txt` не могут попасть в production tree.

## Создать Python environment

Создайте чистое project environment от системного `/usr/bin/python3` и
установите lock-файл из production tree. Владелец target host отвечает за
подходящую версию system Python; project documentation не выбирает versioned
interpreter path.

```bash
/usr/bin/python3 -m venv --clear /opt/transformer/.venv
/opt/transformer/.venv/bin/python -m pip install \
  -r /opt/transformer/requirements.txt
/opt/transformer/.venv/bin/python -m pip check
sudo restorecon -RF /opt/transformer
matchpathcon -V /opt/transformer/.venv/bin/python
```

Окружение создаётся на target host и не копируется с другой системы. После
смены system `/usr/bin/python3` его всегда пересоздают этим же сценарием.
Application dependencies не устанавливаются в system Python или user-site.

## Проверить runtime до unit

Все команды ниже должны завершиться успешно до создания и включения unit:

```bash
test -x /opt/transformer/.venv/bin/python
test -f /opt/transformer/app/main.py
test -f /opt/transformer/.env
/opt/transformer/.venv/bin/python /opt/transformer/app/main.py --help
/opt/transformer/.venv/bin/python -c \
  'import alembic, dotenv, jsonschema, psycopg, pyarrow, sqlalchemy, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits
```

Если здесь возникает ошибка, unit запускать ещё рано. В частности, успешный
`pip check` не заменяет import-проверку обязательных application dependencies.

## Применить миграции

Сначала проверьте, затем явно примените migrations и повторите проверку:

```bash
/opt/transformer/.venv/bin/python \
  /opt/transformer/app/main.py db migrations status
/opt/transformer/.venv/bin/python \
  /opt/transformer/app/main.py db migrations apply
/opt/transformer/.venv/bin/python \
  /opt/transformer/app/main.py db migrations status
```

Продолжать можно только когда `status` больше не сообщает `Pending migrations:
yes`. Миграции не выполняются при запуске service. Если первая команда не
может подключиться к PostgreSQL, сначала нужно восстановить database
connectivity или проверить `/opt/transformer/.env`.

## Создать unit-файл

```bash
sudo nano /etc/systemd/system/transformer.service
```

Содержимое файла:

```ini
[Unit]
Description=Transformer Arrow Flight service
Documentation=file:/opt/transformer/docs/deployment/systemd.md
After=network-online.target
Wants=network-online.target

# Не допускают restart-loop, если production tree ещё не подготовлен.
ConditionPathExists=/opt/transformer/app/main.py
ConditionPathIsExecutable=/opt/transformer/.venv/bin/python
StartLimitIntervalSec=60s
StartLimitBurst=5

[Service]
Type=exec

User=nerv
Group=nerv
WorkingDirectory=/opt/transformer

Environment=PYTHONUNBUFFERED=1

ExecStart=/opt/transformer/.venv/bin/python /opt/transformer/app/main.py flight serve --host=0.0.0.0 --port=8815 --allow-plaintext

Restart=on-failure
RestartSec=5s

KillMode=mixed
TimeoutStopSec=60s
UMask=0077

StandardOutput=journal
StandardError=journal
SyslogIdentifier=transformer

[Install]
WantedBy=multi-user.target
```

`ConditionPath*` предотвращают выполнение `ExecStart`, когда production tree
неполный. `StartLimit*` останавливают бесконечные перезапуски после пяти
неудачных запусков за минуту. Это не заменяет preflight выше: conditions не
проверяют зависимости или миграции.

Не добавляйте `EnvironmentFile`. Приложение самостоятельно читает ровно один
файл `/opt/transformer/.env` после запуска process от имени `nerv`; это
исключает второй источник PostgreSQL configuration в unit.

`--allow-plaintext` допустим только для доверенной private network. Bearer
authentication остаётся обязательной; TLS/mTLS настраиваются явно по
[Flight runbook](../flight-operations.md).

## Запустить сервис

```bash
sudo systemd-analyze verify /etc/systemd/system/transformer.service
sudo systemctl daemon-reload
sudo systemctl reset-failed transformer
sudo systemctl enable --now transformer
systemctl is-active --quiet transformer
ss -ltn 'sport = :8815'
```

При ошибке не оставляйте unit в restart-loop. Сначала посмотрите статус и
журнал текущей загрузки:

```bash
systemctl status transformer --no-pager -l
journalctl -u transformer -b --no-pager -n 100
```

Следить за журналом:

```bash
journalctl -u transformer -f
```

## Открыть порт 8815

Если используется `firewalld` и endpoint действительно должен быть доступен
из доверенной сети:

```bash
sudo firewall-cmd --permanent --add-port=8815/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

Подробности о PostgreSQL, TLS, recovery и runtime storage приведены в
[Flight runbook](../flight-operations.md).
