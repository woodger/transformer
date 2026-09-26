# ADR 0033: отдельный бинарный BCE с весом положительного класса

- Статус: Принято
- Дата решения: 2026-09-24

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

Обычный `BinaryCrossEntropyWithLogits` допускает дробные метки в `[0, 1]`.
Добавление к нему параметра веса положительного класса изменило бы допустимую
область даже при значении веса `1`. Для контролируемого обучения на редком
строго бинарном событии нужен явный provider-executed primitive без
дублирования строк, повторной выборки и зависимости от непрозрачной identity
цели.

## Решение

Принять отдельный прямой примитив
`PositiveClassWeightedBinaryCrossEntropyWithLogits` с обязательным конечным
`positiveClassWeight > 0`.

Он допустим только для строго бинарного target slot и не участвует во
вспомогательных компонентах первой ревизии. Публичное prediction выводится
из raw logit как `sigmoid(logit - log(positiveClassWeight))`; это вероятность
для невзвешенного распределения представленных строк, а не обещание
обобщающей статистической калибровки.

Изменение выпускается destructive clean cut через Semantic v4 и Flight v17.
Существующие generations не конвертируются и не читаются новой границей.

## Рассмотренные альтернативы

- Добавить `positiveClassWeight` существующему BCE. Отклонено: смешало бы
  soft-label и binary-only domains.
- Выводить вес по target identity или составу загрузки. Отклонено: скрыло бы
  семантику и перенесло бы политику вызывающей системы в Transformer.
- Возвращать взвешенный score без поправки. Отклонено: он не является
  вероятностью исходного распределения строк.

## Последствия

- Вес класса и его изменение входят в objective/model compatibility identity.
- Telemetry считает target MAE/RMSE по скорректированному public prediction,
  но direct и total losses остаются наблюдениями взвешенной objective.
- Model Catalog, predict-create и Model Topology раскрывают только
  checkpoint-owned разрешённую проекцию и выводимую поправку.
- Физическая Arrow плоскость и `indexedFeatureBlocks` не меняются.

## Текущая документация

- [Семантический контракт v4](../../app/contracts/semantic/v4/README.md)
- [Контракт Arrow Flight v17](../../app/contracts/flight/v17/README.md)
- [Контракт Model Catalog v4](../../app/contracts/model_catalog/v4/README.md)
- [Контракт Model Topology v2](../../app/contracts/model_topology/v2/README.md)
- [Справочник objective и loss](../losses.md)
