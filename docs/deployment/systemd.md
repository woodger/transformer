# Развертывание через systemd

> Type: Reference. Production-запуск `transformer flight serve` через systemd.

Service запускается из `/home/nerv/transformer` от `nerv:nerv`.
Project `.venv`, `.env` и PostgreSQL migrations должны быть подготовлены до
включения unit. Правила окружения находятся в
[политике Python runtime](../policy/python-runtime-policy.md), а параметры
Flight service — в [Flight runbook](../flight-operations.md).

## Создать unit-файл

Service запускает project interpreter напрямую с тем же command path, который
используется для локального CLI.

```bash
sudo nano /etc/systemd/system/transformer.service
```

Содержимое файла:

```ini
[Unit]
Description=Transformer Arrow Flight service
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=nerv

ExecStart=/home/nerv/transformer/.venv/bin/python /home/nerv/transformer/app/main.py flight serve --host=0.0.0.0 --port=8815 --allow-plaintext

Restart=on-failure
RestartSec=5

KillMode=mixed
TimeoutStopSec=60
UMask=0077

[Install]
WantedBy=multi-user.target
```

## Запустить сервис

```bash
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
