# Развёртывание через systemd

> Type: Reference. Production-запуск `transformer flight serve` через systemd.

Production-проект установлен в `/opt/transformer` и запускается от
`nerv:nerv`. Рабочая копия `/home/nerv/transformer` не является частью
production runtime. Целевая система — Fedora с SELinux в режиме `Enforcing`.

## Подготовить проект

Первичный перенос выполняется при остановленном сервисе. Он переносит `.env`,
`models/` и `recovery/`, но исключает старое Python-окружение и кэши:

```bash
sudo systemctl stop transformer
test ! -e /opt/transformer
sudo install -d -o nerv -g nerv -m 0755 /opt/transformer

tar \
  --exclude=.git \
  --exclude='.venv*' \
  --exclude=.cache \
  --exclude=.pytest_cache \
  --exclude=.ruff_cache \
  --exclude=transformer-pending.service \
  --exclude=transformer.service.pending \
  -C /home/nerv/transformer -cf - . | \
  tar -C /opt/transformer -xf -

chmod 0600 /opt/transformer/.env
```

Создать проектное окружение Python 3.11 и установить зафиксированные
зависимости:

```bash
/usr/bin/python3.11 -m venv /opt/transformer/.venv
/opt/transformer/.venv/bin/python -m pip install \
  -r /opt/transformer/requirements.txt
/opt/transformer/.venv/bin/python -m pip check
sudo restorecon -RF /opt/transformer
matchpathcon -V /opt/transformer/.venv/bin/python
```

Окружение создаётся на целевом сервере и не копируется с другой системы.
`requirements.txt` — единственный источник версий Python-пакетов проекта.
Production root находится под `/opt`, потому что SELinux запрещает systemd
исполнять файлы с типом `user_home_t` из `/home`.

Проверить CUDA:

```bash
/opt/transformer/.venv/bin/python -c \
  'import alembic, dotenv, psycopg, pyarrow, sqlalchemy, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits
```

Если хотя бы одна команда завершается с ошибкой, unit запускать ещё рано.

## Применить миграции

```bash
/opt/transformer/.venv/bin/python \
  /opt/transformer/app/main.py db migrations status
/opt/transformer/.venv/bin/python \
  /opt/transformer/app/main.py db migrations apply
```

Миграции не выполняются при запуске сервиса. Если команда `status` не может
подключиться к PostgreSQL, сначала нужно восстановить доступность базы данных.

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

[Service]
Type=exec

User=nerv
Group=nerv
WorkingDirectory=/opt/transformer

Environment=PYTHONUNBUFFERED=1

ExecStart=/opt/transformer/.venv/bin/python /opt/transformer/app/main.py flight serve --host=0.0.0.0 --port=8815 --allow-plaintext

Restart=on-failure
RestartSec=5

KillMode=mixed
TimeoutStopSec=60
UMask=0077

StandardOutput=journal
StandardError=journal
SyslogIdentifier=transformer

[Install]
WantedBy=multi-user.target
```

Не добавляйте `EnvironmentFile`. Приложение самостоятельно читает ровно один
файл `/opt/transformer/.env` после запуска process от имени `nerv`; это
исключает второй источник PostgreSQL configuration в unit.

## Запустить сервис

```bash
sudo systemd-analyze verify /etc/systemd/system/transformer.service
sudo systemctl daemon-reload
sudo systemctl reset-failed transformer
sudo systemctl enable --now transformer
```

Проверить статус:

```bash
systemctl status transformer --no-pager -l
```

Посмотреть журнал текущей загрузки:

```bash
journalctl -u transformer -b --no-pager -n 100
```

Следить за журналом:

```bash
journalctl -u transformer -f
```

Проверить, что сервис слушает порт `8815`:

```bash
ss -ltn 'sport = :8815'
```

## Открыть порт 8815

Если используется `firewalld`:

```bash
sudo firewall-cmd --permanent --add-port=8815/tcp
sudo firewall-cmd --reload
```

Проверить, что порт открыт:

```bash
sudo firewall-cmd --list-ports
```

Подробности о PostgreSQL, TLS, recovery и runtime storage приведены в
[`../flight-operations.md`](../flight-operations.md).
