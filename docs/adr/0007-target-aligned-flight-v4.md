# ADR 0007: target-aligned public ML contract

- Status: Accepted
- Decision date: 2026-08-12

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Target Inventory и public prediction Transformer имели одинаковую ширину, но
разную семантику отдельных координат. Consumer мог сравнивать значения по
одинаковому index, хотя model output представлял raw logits или внутреннюю
Gaussian uncertainty вместо соответствующего target.

Это создавало скрытый mapping между training supervision, checkpoint,
prediction и downstream evaluation. Исправление observable semantics нельзя
было вносить под прежней protocol identity.

## Решение

Public prediction совпадает с target-space по ширине, порядку, семантике и
range каждой координаты. Каждая public coordinate имеет прямой supervised
loss. Raw logits и отдельная Gaussian uncertainty head остаются private model
details и преобразуются либо исключаются до public boundary.

Quality metrics считаются по каждой target coordinate отдельно. Checkpoint
selection использует только непосредственно supervised components, а
auxiliary losses не меняют public semantics. Полная objective configuration
получает canonical digest и входит в compatibility identity checkpoint,
recovery и published model.

Изменение выполнено как breaking protocol/model contract boundary без aliases
и скрытой интерпретации прежних checkpoints. Последующее изменение написания
public identities в ADR 0015 сохранило сам target-aligned принцип.

## Рассмотренные альтернативы

- Оставить raw model outputs и выполнять mapping в Consumer. Отклонено из-за
  двух владельцев semantic contract и риска сравнения несовместимых значений.
- Сохранить прежний schema identifier и изменить только интерпретацию.
  Отклонено как скрытая breaking change.
- Использовать auxiliary loss вместо direct supervision некоторых public
  heads. Отклонено: каждая опубликованная coordinate должна иметь собственный
  training signal.
- Сводить качество к одной MAE/MSE по разнородным coordinates. Отклонено как
  потеря предметной семантики.

## Последствия

- Consumer может сопоставлять `prediction[i]` и `target[i]` без локального
  semantic mapping.
- Objective, checkpoint, recovery и model metadata образуют одну compatibility
  boundary.
- Изменение public target semantics требует новой contract version и
  переобучения несовместимых models.

## Текущая документация

- [Контракт Arrow Flight v6](../../app/contracts/flight/v6/README.md)
- [Функция потерь](../losses.md)
- [Training runtime](../training-runtime.md)
- [ADR 0015: единая identity индикаторов](0015-unified-indicator-identity-flight-v5.md)
