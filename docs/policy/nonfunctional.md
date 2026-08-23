# Нефункциональные требования

> Type: Policy. Этот документ фиксирует свойства Transformer, которые нельзя
> нарушать рабочей правкой.

## Корректность и воспроизводимость

- заданные seed и deterministic mode сохраняют заявленную семантику;
- изменение размера transport payload не меняет training trajectory;
- optimizer, scheduler, early stopping и best checkpoint принадлежат всему
  training job, а не отдельному Flight payload;
- явный `cuda` не подменяется CPU;
- shape, dtype, NaN/Infinity и masking semantics валидируются до вычисления.

## Контракты данных

- Arrow schema и границы logical payload сохраняются;
- binary stdout не смешивается с diagnostics;
- checkpoint metadata достаточно для совместимого prediction;
- опубликованные модели неизменяемы и появляются только после успешного fit;
- Flight idempotency не приводит к повторному запуску Torch execution.

## Надёжность и хранение

- PostgreSQL остаётся единственным durable control-plane source of truth;
- prediction payload и незавершённые attempt artifacts остаются в
  `/tmp/transformer`;
- fit payload и completed-epoch recovery checkpoints остаются в project
  `recovery/`;
- успешно опубликованные checkpoints остаются в project `models/`;
- при успешном best-effort сборе immutable training metrics artifact живёт
  вместе с model generation, а его OpenSearch projection доставляется через
  PostgreSQL outbox;
- потеря runtime storage инвалидирует prediction jobs, но не fit с целыми
  persistent recovery artifacts;
- fit возобновляется только с зарегистрированной границы полной глобальной
  эпохи, без silent restart при повреждении recovery;
- файловая публикация и database transitions остаются атомарными;
- runtime и recovery directories принадлежат одному процессу сервиса;
- CUDA attempt привязан к одному physical GPU, а подтверждённо потерянный GPU
  не возвращается в pool до следующего Linux boot.

## Безопасность и эксплуатация

- bearer authentication действует для каждого Flight transport; owner identity
  равна точному subject PostgreSQL-backed API token;
- plaintext разрешается только явно;
- сбой сбора, конфигурации или доставки telemetry не меняет результат fit,
  model lifecycle, startup или shutdown; transport profile определяется
  deployment policy;
- paths и subprocess arguments не принимаются из network request произвольно;
- service не опрашивает PostgreSQL на каждом RPC или в idle worker loop;
- SIGTERM, cancellation и process-group cleanup не оставляют активных workers;
- CLI, environment names, логирование и exit behavior меняются только
  намеренно как публичный контракт.

Изменение, которое выдаёт правильный happy-path результат, но нарушает одно из
этих свойств, не считается корректным.
