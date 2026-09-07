# ADR 0021: provider-neutral public GPU device interface

- Status: Accepted
- Decision date: 2026-08-28

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Публичные local fit/predict CLI и Arrow Flight interfaces использовали
значение `cuda` для выбора accelerator device. Это связывало Consumer-ов с
конкретным runtime provider, хотя CUDA является деталью текущей реализации
Transformer. Термин также расходился с пользовательским понятием GPU resource,
которое требуется для scheduling, capacity и диагностики независимо от
backend-а.

Изменение затрагивает wire schemas, descriptors, capabilities, health, errors,
metrics и local CLI. Одновременная поддержка старого и нового spelling создала
бы два имени одной semantic identity.

## Решение

Публичной device identity является `gpu`. Local fit/predict CLI и Arrow Flight
принимают только `cpu`, `gpu` и `auto`; `cuda` не является публичным alias.

CUDA остаётся внутренним execution backend. Adapter boundary преобразует
public `gpu` во внутренний `cuda` и возвращает наружу `gpu`. Worker protocol,
persistence и backend diagnostics могут использовать CUDA-specific terms,
поскольку они не являются Consumer API.

Breaking wire change выпускается как Flight v6. Flight v5 удаляется без
parallel runtime, aliases или compatibility layer.

## Рассмотренные альтернативы

- Сохранить `cuda` в публичном API. Отклонено, потому что provider-owned name
  становится долгосрочным Consumer contract.
- Принимать одновременно `gpu` и `cuda`. Отклонено: aliases сохраняют две
  semantic identity и усложняют validation и telemetry.
- Переименовать CUDA во всех внутренних слоях. Отклонено: persistence, worker
  protocol и backend code корректно описывают фактическую реализацию, а их
  изменение не требуется для public boundary.
- Поддерживать Flight v5 и v6 параллельно. Отклонено, поскольку для перехода
  предоставлено согласованное maintenance window и active v5 jobs отсутствуют.

## Последствия

- Consumer-ы зависят от логического GPU resource, а не от runtime provider.
- Смена accelerator backend-а не требует нового public device spelling.
- Consumer-ы должны перейти на Flight v6 и заменить `cuda` на `gpu` без
  compatibility period.
- Внутренние device values в persistence и worker protocol не мигрируют и
  преобразуются только на adapter boundary.
- Переименованные operational metrics начинают новые series; исторические
  series не переписываются.
- Checkpoint и model artifact contracts не меняются, поэтому принудительное
  переобучение моделей не требуется.
- Operator diagnostics и worker failures могут по-прежнему называть CUDA.

## Текущая документация

- [Контракт Arrow Flight v13](../../app/contracts/flight/v13/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Local CLI](../cli/index.md)
- [Архитектура](../architecture.md)
