<!--
CONTRACT DOCUMENT — SEMANTIC CHANGE MODE

Этот файл фиксирует проверенный deployment contract и не является шаблоном,
кандидатом на cleanup или stylistic rewrite.

Менять его можно только по явному требованию текущей задачи и только в
минимальном фрагменте, необходимом для изменения названного контракта. До
изменения нужно определить точный contract delta и сохранить все независимые
инварианты. Сходство с другим проектом не разрешает переписывать документ
целиком, удалять соседние положения или переносить его deployment-детали.

После изменения нужно проверить semantic diff и явно сообщить, какие свойства
контракта изменены, а какие сохранены.
-->

# Развертывание через systemd

> Type: Reference. Production-запуск `transformer flight serve` через systemd.

Service запускается из `/home/nerv/transformer` от `nerv:nerv`.
Project `.venv`, `.env` и
[PostgreSQL migrations](../operations/database-migrations.md) должны быть
подготовлены до включения unit. Правила окружения находятся в
[политике Python runtime](../policy/python-runtime-policy.md), а параметры
Flight service — в [Flight runbook](../operations/flight-service.md).
Необязательная доставка training metrics настраивается отдельно по
[инструкции OpenSearch](opensearch.md).

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
WorkingDirectory=/home/nerv/transformer

ExecStart=/home/nerv/transformer/.venv/bin/python app/main.py flight serve --host=0.0.0.0 --port=8815

Restart=on-failure
RestartSec=5

KillMode=mixed
TimeoutStopSec=60
UMask=0077

[Install]
WantedBy=multi-user.target
```

## Настроить SELinux

Разрешить systemd читать Python-ссылки виртуального окружения:

```bash
sudo semanage fcontext -a -f l -t bin_t \
  '/home/nerv/transformer/\.venv/bin/python([0-9]+(\.[0-9]+)?)?'
sudo restorecon -Rv /home/nerv/transformer/.venv/bin
```

Проверить тип ссылок:

```bash
ls -lZ /home/nerv/transformer/.venv/bin/python*
```

Python-ссылки должны иметь тип `bin_t`. После пересоздания `.venv` повторно
выполнить `restorecon`.

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
