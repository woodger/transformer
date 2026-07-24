# Развёртывание через systemd

> Type: Reference. Production-запуск `transformer flight serve` через systemd.

Целевой проект установлен в `/home/nerv/transformer`, сервис работает от
`nerv:nerv` через `/usr/bin/python3.11`. Развёртывание выполняется вручную.
Подробная семантика Flight service описана в
[`../flight-operations.md`](../flight-operations.md).

## Подготовить окружение

Используйте корневой файл [`.env.example`](../../.env.example) как образец для
`/home/nerv/transformer/.env`. Не перезаписывайте уже настроенный файл с
доступами к PostgreSQL.

Параметры Transformer service задаются в `/home/nerv/transformer/app/config.py`.
Проверьте их перед первым запуском.
TLS или mTLS включается только явными certificate options в `ExecStart`.

```bash
sudo chown nerv:nerv /home/nerv/transformer/.env
sudo chmod 0600 /home/nerv/transformer/.env

sudo install -d -o nerv -g nerv -m 0700 \
  /home/nerv/transformer/models \
  /home/nerv/transformer/recovery

sudo -u nerv -H /usr/bin/python3.11 -m pip check
```

## Подготовить runtime-директорию

```bash
sudo nano /etc/tmpfiles.d/transformer.conf
```

Содержимое файла:

```text
d /tmp/transformer 0700 nerv nerv -
```

Применить конфигурацию:

```bash
sudo systemd-tmpfiles --create /etc/tmpfiles.d/transformer.conf
stat -c '%U:%G %a %n' /tmp/transformer
```

Ожидаемый результат:

```text
nerv:nerv 700 /tmp/transformer
```

Потеря `/tmp/transformer` инвалидирует prediction jobs и незавершённые attempt
artifacts. Fit jobs продолжаются по данным из
`/home/nerv/transformer/recovery`; успешно опубликованные модели остаются в
`/home/nerv/transformer/models`.
Если системная temporary directory отличается от `/tmp`, соответствующий
вычисляемый путь необходимо указать и в tmpfiles configuration.

## Создать unit-файл

```bash
sudo nano /etc/systemd/system/transformer.service
```

Содержимое файла:

```ini
[Unit]
Description=Transformer Arrow Flight service
Documentation=file:/home/nerv/transformer/docs/deployment/systemd.md
Wants=network-online.target
After=network-online.target systemd-tmpfiles-setup.service
StartLimitIntervalSec=60s
StartLimitBurst=5

[Service]
Type=exec
User=nerv
Group=nerv
WorkingDirectory=/home/nerv/transformer
EnvironmentFile=/home/nerv/transformer/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3.11 /home/nerv/transformer/app/main.py flight serve

Restart=on-failure
RestartSec=5s

KillSignal=SIGTERM
KillMode=mixed
SendSIGKILL=yes
TimeoutStopSec=60s

UMask=0077
PrivateTmp=no
PrivateDevices=no
ProtectSystem=full

StandardOutput=journal
StandardError=journal
SyslogIdentifier=transformer

[Install]
WantedBy=multi-user.target
```

## Применить миграции

```bash
sudo -u nerv -H /usr/bin/python3.11 \
  /home/nerv/transformer/app/main.py db migrations status

sudo -u nerv -H /usr/bin/python3.11 \
  /home/nerv/transformer/app/main.py db migrations apply
```

Миграции не выполняются автоматически при запуске сервиса.

## Запустить сервис

```bash
sudo systemd-analyze verify /etc/systemd/system/transformer.service
sudo systemctl daemon-reload
sudo systemctl enable --now transformer
```

Проверить статус:

```bash
systemctl status transformer
```

Посмотреть логи:

```bash
journalctl -u transformer -f
```

Проверить CUDA от имени service user:

```bash
sudo -u nerv -H /usr/bin/python3.11 -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

sudo -u nerv -H nvidia-smi \
  --query-gpu=index,uuid,name \
  --format=csv,noheader,nounits
```

При штатном перезапуске сервис сначала использует drain interval. Если fit всё
же прерывается, следующая попытка продолжает его с последней полностью
завершённой глобальной эпохи; незавершённая эпоха выполняется повторно.
Подтверждённо отказавший GPU записывается в
`/tmp/transformer/cuda-quarantine.json`: `systemctl restart` не возвращает его
в scheduler, а новая загрузка Linux очищает boot-scoped quarantine.

## Открыть порт 8815

Этот шаг нужен, только если `HOST_DEFAULT` в `app/config.py` или переданный
через `--host` адрес не является loopback-адресом и к сервису подключается
удалённый Inventory.

Если используется `firewalld`:

```bash
sudo firewall-cmd --permanent --add-port=8815/tcp
sudo firewall-cmd --reload
```

Проверить, что порт открыт:

```bash
sudo firewall-cmd --list-ports
```
