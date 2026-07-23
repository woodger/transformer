# Развёртывание Transformer через systemd

Этот документ фиксирует deployment contract для production-хоста:

- Fedora с systemd 259;
- проект в `/home/nerv/transformer`;
- процесс работает как `nerv:nerv`;
- Python — `/usr/bin/python3.11`;
- runtime payloads — `/tmp/transformer`;
- опубликованные модели — `/home/nerv/transformer/models`;
- PostgreSQL и Flight configuration — `/home/nerv/transformer/.env`;
- CUDA доступна процессу через host NVIDIA driver.

Systemd управляет только жизненным циклом процесса. PostgreSQL migrations и API
access tokens остаются явными операторскими командами.

## Предварительная проверка

Все runtime dependencies должны быть установлены для Python 3.11 пользователя
`nerv`. Проверка не должна выполняться через root Python environment:

```bash
sudo -u nerv -H /usr/bin/python3.11 -m pip check
sudo -u nerv -H /usr/bin/python3.11 - <<'PY'
import alembic
import numpy
import psycopg
import pyarrow
import sqlalchemy
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

`cuda False` блокирует только CUDA jobs, но для указанного GPU-хоста это
считается ошибкой deployment и должно быть исправлено до запуска сервиса.

Подготовьте persistent model directory:

```bash
sudo install -d -o nerv -g nerv -m 0700 \
  /home/nerv/transformer/models
```

## Environment file

Unit читает `/home/nerv/transformer/.env`. Существующий файл с PostgreSQL
credentials не нужно заменять: дополните его service settings из
[`transformer.env.example`](../deploy/systemd/transformer.env.example).

Минимальный plaintext-вариант для loopback:

```dotenv
TRANSFORMER_RUNTIME_DIR=/tmp/transformer
TRANSFORMER_HOST=127.0.0.1
TRANSFORMER_PORT=8815
TRANSFORMER_ALLOW_PLAINTEXT=true
```

Inventory на другом сервере не сможет подключиться к loopback. Для такого
deployment задайте точный LAN address и выбранную transport policy. TLS
certificate и key должны быть доступны пользователю `nerv`.

Ограничьте доступ к файлу:

```bash
sudo chown nerv:nerv /home/nerv/transformer/.env
sudo chmod 0600 /home/nerv/transformer/.env
```

Токены в `.env` не записываются: они выпускаются и отзываются через PostgreSQL
командами `auth tokens`.

## Установка unit и runtime directory policy

Установите unit и tmpfiles configuration из checkout:

```bash
cd /home/nerv/transformer

sudo install -m 0644 \
  deploy/systemd/transformer.service \
  /etc/systemd/system/transformer.service

sudo install -m 0644 \
  deploy/systemd/transformer.tmpfiles.conf \
  /etc/tmpfiles.d/transformer.conf

sudo systemd-tmpfiles --create /etc/tmpfiles.d/transformer.conf
sudo systemd-analyze verify /etc/systemd/system/transformer.service
sudo systemctl daemon-reload
```

Ожидаемые параметры runtime directory:

```bash
stat -c '%U:%G %a %n' /tmp/transformer
```

Результат должен быть:

```text
nerv:nerv 700 /tmp/transformer
```

`PrivateTmp=no` в unit является частью storage contract. Обычный
`systemctl restart` должен видеть прежний `storage-epoch` и spool. Потеря
смонтированного в RAM `/tmp` после полного сбоя или reboot, напротив, создаёт
новый epoch и инвалидирует все jobs.

## PostgreSQL schema и API access

До первого запуска проверьте и явно примените migrations:

```bash
sudo -u nerv -H /usr/bin/python3.11 \
  /home/nerv/transformer/app/main.py db migrations status

sudo -u nerv -H /usr/bin/python3.11 \
  /home/nerv/transformer/app/main.py db migrations apply
```

Не добавляйте `db migrations apply` в `ExecStartPre`: изменение схемы является
отдельной deployment-операцией. Сам сервис откажется запускаться, если schema
revision не совпадает с Alembic head.

Если access token ещё не выпущен:

```bash
sudo -u nerv -H /usr/bin/python3.11 \
  /home/nerv/transformer/app/main.py \
  auth tokens issue --subject=inventory-production
```

Сохраните напечатанный credential на стороне Inventory. Повторно получить его
через `auth tokens list` нельзя.

## Запуск и наблюдение

```bash
sudo systemctl enable --now transformer.service
systemctl status transformer.service
journalctl -u transformer.service -f
```

Unit использует `Type=exec`. Состояние `active` подтверждает, что основной
процесс запущен, но application readiness проверяется authenticated Flight
action `transformer.v1.health`.

Автоматический restart выполняется только после failure:

```ini
Restart=on-failure
RestartSec=5s
```

Обычный `systemctl stop` не запускает сервис повторно.

## Stop и restart semantics

Unit использует `KillMode=mixed`: первоначальный SIGTERM получает только
основной процесс. Transformer закрывает queue-claim boundary, переходит в
draining и самостоятельно завершает worker process groups. Если процесс не
остановился за 60 секунд, systemd отправляет оставшимся процессам SIGKILL.

Текущая application policy резервирует:

- 30 секунд на drain running workers;
- 10 секунд между worker SIGTERM и SIGKILL;
- 20 секунд запаса на остановку Flight, maintenance, token listener и
  PostgreSQL pool.

Если `TRANSFORMER_SHUTDOWN_DRAIN_SECONDS` или
`TRANSFORMER_CANCEL_GRACE_SECONDS` увеличиваются, одновременно увеличьте
`TimeoutStopSec`.

Не выполняйте restart во время долгого running fit, если не готовы отменить
его. `UPLOADING`, `SEALED` и `QUEUED` jobs переживают обычный restart при
сохранённом `/tmp`; незавершённый `RUNNING` fit автоматически не продолжается.

## Обновление

Перед обновлением:

1. убедитесь, что restart допустим для текущих jobs;
2. обновите checkout и Python dependencies;
3. выполните `db migrations status`, затем при необходимости
   `db migrations apply`;
4. повторно установите unit и tmpfiles configuration;
5. выполните `systemd-analyze verify`, `daemon-reload` и restart;
6. проверьте journal и authenticated Flight health.

```bash
sudo systemctl daemon-reload
sudo systemctl restart transformer.service
```

## CUDA и ограничения sandbox

Unit намеренно не включает `PrivateDevices`, `ProtectProc` и PID namespace
isolation:

- PyTorch должен видеть NVIDIA device nodes;
- startup recovery должен читать `/proc/<pid>/stat`;
- worker subprocesses должны оставаться в service cgroup и в том же PID
  namespace, что и основной процесс.

После установки unit проверьте CUDA именно от имени service user:

```bash
sudo -u nerv -H /usr/bin/python3.11 - <<'PY'
import torch

assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

На Fedora с SELinux не отключайте enforcement при ошибке запуска. Сначала
проверьте journal и AVC events:

```bash
journalctl -u transformer.service
sudo ausearch -m AVC -ts recent
```

## Удаление unit

Остановка и удаление unit не должны удалять PostgreSQL schema, access tokens
или `/home/nerv/transformer/models`:

```bash
sudo systemctl disable --now transformer.service
sudo rm /etc/systemd/system/transformer.service
sudo rm /etc/tmpfiles.d/transformer.conf
sudo systemctl daemon-reload
```

Удаление `/tmp/transformer` является явной потерей runtime storage и при
следующем запуске инвалидирует все связанные jobs.
