# Changelog

Все заметные изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
а проект следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-07-22

### Changed

- CLI-параметр адреса Flight service переименован из `--bind-host` в `--host`,
  JSON-поле — из `bindHost` в `host`, а environment variable — из
  `TRANSFORMER_FLIGHT_BIND_HOST` в `TRANSFORMER_FLIGHT_HOST`; прежние имена
  больше не поддерживаются. Внутренняя конфигурация также использует единую
  пару `host` / `port`.
- Глобальный CLI help сокращён до global options и списка команд; примеры
  перенесены в применимые command-specific help, а для `--version` добавлена
  короткая форма `-v`.
- Команда запуска Flight service переименована из `serve-flight` в составную
  `flight serve`; прежнее имя больше не поддерживается.

### Fixed

- Flight fit теперь выполняет настроенное число эпох над всем sealed input job:
  внутри каждой эпохи durable payloads читаются по ordinal, а loss schedule,
  optimizer, checkpoint selection и early stopping больше не перезапускаются
  на границах транспортных payloads.

## [0.1.2] - 2026-07-19

### Changed

- Упрощены инструкции по установке PyTorch, NumPy, PyArrow и pytest на
  хосте; виртуальное окружение больше не представлено как обязательное.
- Главный README сфокусирован на настройке и запуске Transformer без
  consumer-specific сценариев.

### Removed

- Удалён временный `requirements.txt`; базовая установка зависимостей
  описана непосредственно в README.

### Fixed

- Flight v1 status для `CANCELLED` job теперь возвращает `error: null`, как
  требует нормативная JSON Schema. Старые durable-записи с cancellation error
  также сериализуются в wire-valid status без удаления state directory.

## [0.1.1] - 2026-07-19

### Added

- Добавлен single-instance Arrow Flight v1 job service для remote fit и predict:
  durable SQLite/WAL ledger, filesystem spool, очереди, subprocess workers, cancellation,
  recovery, retention и operational observability.
- Добавлен нормативный Flight contract v1 с JSON Schemas, golden JSON/Arrow fixtures,
  ADR и production runbook.
- CLI переведён на настоящие mode-specific subcommands `fit`, `predict`,
  `fit-stream`, `predict-stream` и `plot-metrics` с ранней валидацией аргументов.
- Checkpoint v2 сохраняет model, training и data-schema configuration; чтение
  wrapped v1 и raw legacy checkpoints сохранено.
- Добавлены reproducibility controls `--seed` и `--deterministic`, атомарная
  публикация артефактов и ограничение размера stream frame.
- Добавлен monitor `ret_mae_skill` относительно zero-return baseline, выбор
  best checkpoint и настраиваемый minimum improvement.

### Changed

- Arrow contract уточняет типы, ширину, null/NaN/Infinity semantics, диапазоны
  target values и typed-empty input/output.
- Device policy унифицирована до `auto`, `cpu` и `cuda`; явно запрошенная CUDA
  больше не подменяется CPU.
- Streaming protocol фиксирует terminator/EOF semantics, binary-only stdout для
  `predict-stream` и один semantic frame на один logical payload.
- Поля metrics JSONL получили однозначные имена (`loss_ret`, `loss_prob`,
  `loss_ev`, `loss_vol`, `grad_norm` и расширенные timing/ratio names), а non-finite
  значения сериализуются как JSON `null`.
- Training console output разделён на одну строку конфигурации и компактную
  epoch summary.

### Removed

- Удалены legacy aliases `--amp` и `--per-week`; поддерживаемые имена —
  `--use-amp`, `--loss-schedule` и `--stage-size`.

### Fixed

- Gaussian NLL переведён на численно устойчивую формулу через
  `var = sigma² + 1e-6`.
- Loss schedule для step mode рассчитывает стадию по global optimizer step.
- Transformer использует последний valid timestep и безопасно обрабатывает
  полностью masked sequences.
- Early stopping корректно отслеживает улучшение monitor до прохождения
  baseline threshold.

### Security

- Flight RPC защищены bearer authentication, subject ownership, TLS/mTLS policy,
  expiring opaque tickets, quotas и запретом arbitrary paths/CLI arguments.

## [0.1.0] - 2026-05-20

### Added

- Первая версия PyTorch Transformer с file-based `fit` и `predict` для Apache Arrow
  datasets.
- Framed Arrow IPC режимы `fit-stream` и `predict-stream` с 8-byte big-endian
  length prefix и обработкой empty frames.
- Staged Bayesian trading loss, early stopping и mixed-precision training на CUDA.
- Training metrics в JSONL и построение SVG-графиков через `plot-metrics`.
- CLI help с описанием data/streaming contracts и команда `--version`.

[Unreleased]: https://github.com/woodger/transformer/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/woodger/transformer/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/woodger/transformer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/woodger/transformer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/woodger/transformer/releases/tag/v0.1.0
