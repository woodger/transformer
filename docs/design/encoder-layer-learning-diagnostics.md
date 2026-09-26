# Диагностика обучения слоёв encoder

> Тип: проектная записка. Решение описывает реализованную диагностику
> локализации сжатия представлений в encoder для Flight v20.

## Контекст

На фиксированном наборе из 1 479 строк одно- и двухслойная модели запоминают
данные, а трёхслойная при тех же входных данных, seed и конфигурации обучения
сводит публичные предсказания почти к константе. Target Head Diagnostics v2
показывает различие построчных состояний после обновления: до encoder, после
каждого его слоя и перед финальной target head, но не показывает, как каждый
слой получает градиент и меняет параметры.

Нельзя объявлять причиной третий слой: все blocks encoder оптимизируются
совместно, а добавленная глубина меняет траекторию ранних слоёв. Также
применённый шаг оптимизатора не доказывает, что каждый слой получил полезный
градиент или заметное изменение параметров.

## Предлагаемая граница

Расширить существующую явно включаемую Target Head Diagnostics до версии 3, а
не создавать отдельный запрос. Наблюдения относятся к той же точной версии
модели, эпохе, закрытому входному артефакту и `representationFlow`, поэтому их
нельзя корректно разъединять или восстанавливать из checkpoint после fit.

Новая настройка исполнения доступна только вместе с уже существующей полной
диагностикой:

```json
{
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact",
  "encoderLayerDiagnostics": "directComponentPerBatch"
}
```

Она входит в `jobConfigSha256` и recovery fence, но не меняет Semantic v4,
`modelDefinitionSha256`, совместимость warm start, objective, публичный
`predict` или плоскость данных Arrow.

Неверная комбинация `encoderLayerDiagnostics` и `targetHead` отклоняется до
создания job как `INVALID_ARGUMENT / INVALID_DIAGNOSTICS_CONFIGURATION`.

## Наблюдения

Для каждого слоя encoder и каждой эпохи Worker собирает три закрытые группы:

- `attention` — параметры self-attention;
- `feedForward` — параметры feed-forward path;
- `normalization` — параметры обеих нормализаций слоя.

Для каждого прямого компонента и группы публикуется градиент до clipping:

```text
gradientL2 = sqrt(Σₚ ||∂c / ∂p||²)
```

где `c` — вклад именованного прямого компонента после его заявленного
`weight`, но до aggregation с другими прямыми и auxiliary components, AMP
scaling, global clipping и `optimizer.step()`. Агрегаты эпохи — среднее,
максимум и число batches с конечным градиентом.

Отдельно для каждой группы публикуется фактическое изменение параметров после
успешного `optimizer.step()`:

```text
parameterUpdateL2 = sqrt(Σₚ ||p_after - p_before||²)
```

Эта величина относится к полному пути оптимизатора: она может включать total
objective, auxiliary components, gradient clipping, Adam и weight decay. Она
не выдаётся как эффект отдельного прямого компонента. Пропущенный AMP update
не входит в её агрегаты; его состояние остаётся в Training Telemetry.

Сырые градиенты, имена и тензоры параметров, строки признаков, байты checkpoint
и сведения об устройстве не пересекают публичную границу.

## Инварианты

- Сбор выполняется только при явном включении и является вспомогательным: его
  ошибка не меняет конечный исход fit.
- `autograd.grad` не записывает `.grad`; сбор не меняет веса, состояние
  optimizer/scaler, траекторию RNG, порядок batches, ModelContract или значения
  предсказаний.
- Отчёт `available` требует плотные observations всех завершённых эпох.
  Частичный артефакт не публикуется как пригодный report.
- V2 артефакты нельзя дополнить ретроспективно. В v3 они дают
  `unavailable / FORMAT_UNSUPPORTED`; двух- и трёхслойные прогоны обучаются
  повторно с новой configuration.
- `representationFlow` сохраняет семантику v2. Его абсолютная L2-величина не
  сама по себе доказывает потерю информации между границами с разной
  normalization; она сопоставляется вместе с градиентами, updates и raw logits.

## Версионная матрица

| Область | Версия | Изменение |
| --- | --- | --- |
| Target Head Diagnostics | v3 | Наблюдения обучения слоёв и действие запроса. |
| Worker | v18 | Новый неизменяемый diagnostics artifact. |
| Checkpoint/recovery | v11 | Настройка исполнения `encoderLayerDiagnostics`. |
| Model Catalog | v6 | Отображение configuration v3 в detail. |
| Flight | v20 | Закрытый набор действий с diagnostics v3 и Catalog v6. |

Semantic v4, Metrics v10, Training Telemetry v4, Model Topology v2, схема
PostgreSQL, схемы Arrow и логическое восстановление `indexedFeatureBlocks` не
меняются. Новая миграция PostgreSQL не нужна.

## Проверяемый результат

Повторные запуски `layers=2` и `layers=3` с одинаковыми frozen input, seed и
остальной configuration должны позволить различить:

1. исчезновение градиентов ранних слоёв;
2. заметные градиенты при малых фактических updates;
3. заметные updates при продолжающемся сжатии forward representation;
4. сосредоточение различий в attention, feed-forward или normalization.

Если градиенты и updates не объяснят сжатие, следующая отдельная задача —
границы внутри encoder block, а не изменение objective или class weighting.

## Результат

Target Head Diagnostics v3, Worker v18, checkpoint/recovery v11, Model Catalog
v6 и Flight v20 реализованы единым переходом. Предыдущий Flight v19 и
опубликованные v2 artifacts не обслуживаются активной runtime-границей.
