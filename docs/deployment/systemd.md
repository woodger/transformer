# Развёртывание через systemd

> Type: Reference. Production-запуск `transformer flight serve` через systemd.

Проект установлен в `/home/nerv/transformer`, сервис запускается от
`nerv:nerv`. Целевая система — Fedora с SELinux в режиме `Enforcing`.

## Подготовить проект

Файл `/home/nerv/transformer/.env` содержит параметры PostgreSQL и читается
самим приложением. Ограничить доступ к нему:

```bash
chmod 0600 /home/nerv/transformer/.env
```

Проверить Python, зависимости и CUDA от имени `nerv`:

```bash
/usr/bin/python3.14 -c \
  'import alembic, dotenv, psycopg, pyarrow, sqlalchemy, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
/usr/bin/python3.14 -m pip check
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits
```

Unit должен использовать тот же интерпретатор, для которого выполнены эти
проверки. Если хотя бы одна команда завершается с ошибкой, unit запускать ещё
рано. Скопированную с другой системы `.venv` использовать нельзя.

## Применить миграции

```bash
cd /home/nerv/transformer
/usr/bin/python3.14 ./app/main.py db migrations status
/usr/bin/python3.14 ./app/main.py db migrations apply
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
Documentation=file:/home/nerv/transformer/docs/deployment/systemd.md
After=network-online.target
Wants=network-online.target

[Service]
Type=exec

User=nerv
Group=nerv
WorkingDirectory=/home/nerv/transformer

Environment=PYTHONUNBUFFERED=1

ExecStart=/usr/bin/python3.14 /home/nerv/transformer/app/main.py flight serve --host=0.0.0.0 --port=8815 --allow-plaintext

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

Не добавляйте `EnvironmentFile=/home/nerv/transformer/.env`. На системе с
SELinux файл проекта имеет тип `user_home_t`, поэтому PID 1 в домене `init_t`
не сможет прочитать его до запуска процесса от имени `nerv`. Отключать SELinux
или создавать правило через `audit2allow` для этого не требуется.

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
