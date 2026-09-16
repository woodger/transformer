# Процессный контракт Transformer Worker v14

> ДОКУМЕНТ КОНТРАКТА. Это внутренний для provider-а процессный протокол между
> сервисом и Worker. Он не является wire contract Consumer.

Worker v14 получает контролируемый сервисом command manifest, выдаёт framed
events и читает/записывает только контролируемые сервисом artifact paths.
Launcher принимает только `--contract-version=14`; reader и alias для прежних
версий Worker отсутствуют.

Manifest содержит валидированное намерение semantic v3, его выпущенные digests,
разрешённую внутреннюю configuration модели, геометрию данных, configuration
training/diagnostics и fences checkpoint/recovery. Из этой информации Worker
материализует принадлежащие Transformer архитектуру и private resources.
Непрозрачные target identities остаются данными для layout и telemetry; ни один
путь исполнения не может ветвиться по имени target-а Consumer.

Внутренние форматы:

```text
transformer-worker protocol 14
transformer-checkpoint-v8
transformer-recovery-v8
```

Recovery проверяет встроенные metadata checkpoint-а, semantic identities,
разрешённую configuration job, input manifest, progress и managed artifact до
загрузки state. Он не переназначает targets, weights или coordinates prediction
между definitions.

Worker потребляет тот же логический input layout indexed feature blocks, что и
Flight v15. Он выводит positions blocks, восстанавливает ограниченные slices в
логический tensor Float32 и выдаёт target-aligned prediction vectors.
Идентификаторы физической Arrow schema остаются процессными деталями
provider-а.

Worker v14 активируется только чистым переходом Flight v15. Существующие
artifacts Worker нельзя возобновить или конвертировать.
