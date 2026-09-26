# Порядок нормализации encoder как явная архитектурная настройка

> Тип: проектная записка. Документ фиксирует согласованную границу
> контролируемого эксперимента pre-norm/post-norm. Это не изменение активных
> Semantic v4, Flight v21 или runtime.

- Статус: согласовано для подготовки канонического пакета
- Срез: 2026-09-26
- Текущая активная граница: Semantic v4, Flight v21, Worker v19,
  checkpoint/recovery v11, Model Catalog v6, Model Topology v2 и Target Head
  Diagnostics v4

## Контекст

На одном frozen input artifact сравнение глубины `2 → 3` локализовало
сжатие различия между строками в encoder path. На epoch 200 у прошедшего L2 и
схлопнувшегося L3 наблюдалось:

| Наблюдение | L2 | L3 |
| --- | ---: | ---: |
| `rawLogit` standard deviation | 2.321 | 0.000010 |
| public prediction standard deviation | 0.181 | 0.000000317 |
| L0 `attentionResidual` flow | 193.1 | 172.6 |
| L0 `norm1` flow | 7.238 | 0.101 |
| flow перед target head | 1.316 | 0.001276 |

Это не устанавливает дефект `LayerNorm`: `rowCenteredL2Mean` измеряет
межстрочное различие, а normalization меняет масштаб. Наблюдение лишь
указывает, что текущая post-norm архитектура становится неустойчивой при
добавлении третьего блока в этой configuration.

Следующий контролируемый A/B обязан менять только порядок нормализации
encoder. Frozen artifact, data binding, seed, batch size, epochs, optimizer,
loss, target, initialization и diagnostics configuration остаются одинаковыми.

## Решение

Новая semantic revision вводит обязательное поле model tuning:

```json
{
  "encoderNormalizationOrder": "postNorm"
}
```

Допустимы только два provider-defined значения:

| Значение | PyTorch mapping | Execution order блока |
| --- | --- | --- |
| `postNorm` | `norm_first=false` | attention → residual → norm1 → feed-forward → residual → norm2 |
| `preNorm` | `norm_first=true` | norm1 → attention → residual → norm2 → feed-forward → residual |

Поле относится к архитектуре, а не к training policy, diagnostics или
transport. В новой closed schema оно обязательно: отсутствие не означает
неявный `postNorm`. Это делает baseline и эксперимент одинаково явными в
exact ModelContract.

Transformer materializes указанное значение в resolved `ModelConfig` и
выпускает новый `modelDefinitionSha256`. Warm start между разными
`encoderNormalizationOrder` запрещён обычной проверкой точного definition.
Нельзя реализовывать это как скрытый CLI/runtime override, глобальную замену
текущего default или post-fit интерпретацию checkpoint.

## Диагностика прямого прохождения

Target Head Diagnostics новой revision сохраняет один closed vocabulary
границ каждого encoder block:

```text
input
norm1
attentionResidual
norm2
feedForwardResidual
```

Каждая величина — `rowCenteredL2Mean` выхода точно названной операции на
одном post-update проходе полного committed artifact в `eval()` и `no_grad()`.
Порядок исполнения определяется `encoderNormalizationOrder` и выдаётся рядом
с массивом layers:

```text
postNorm: input → attentionResidual → norm1 → feedForwardResidual → norm2
preNorm:  input → norm1 → attentionResidual → norm2 → feedForwardResidual
```

`representationFlow.encoderLayers` остаётся финальным output каждого блока,
независимо от порядка normalization. Он остаётся единственной прямой
сопоставимой границей выхода layers; не следует представлять абсолютные
значения разных внутренних boundaries как доказательство потери информации.

Группы `attention`, `feedForward` и `normalization`, direct-component
gradients и parameter updates сохраняют уже принятую semantics. Сбор остаётся
best-effort и не меняет веса, optimizer/scaler state, RNG trajectory, batch
order, selection или prediction values.

## Публичные проекции

Model Catalog detail и summary возвращают exact Semantic v5 `modelTuning` с
`encoderNormalizationOrder`. Predict-create продолжает принимать только
`modelRef` и data binding: порядок normalization уже принадлежит immutable
published generation и не передаётся повторно.

Model Topology новой revision отмечает каждую encoder-layer node тем же
`encoderNormalizationOrder`. Topology не раскрывает module paths, tensors,
параметры, checkpoint или implementation-private execution details.

Training Telemetry не меняется: loss, MAE/RMSE, health, direct/auxiliary
component layout и gradient interactions не приобретают новую semantics.

## Versioning и миграция

Предлагаемая матрица:

| Область | Подготовленная revision | Причина |
| --- | --- | --- |
| Semantic | v5 | Closed `modelTuning` содержит обязательный порядок normalization. |
| Flight | v22 | Новый closed action/capability surface с Semantic v5. |
| Worker | v20 | Resolved `ModelConfig`, manifest и artifact format. |
| checkpoint/recovery | v12 | Persisted `ModelConfig` и recovery fence. |
| Model Catalog | v7 | Detail/summary с Semantic v5 и checkpoint v12. |
| Model Topology | v3 | Layer nodes раскрывают порядок normalization. |
| Target Head Diagnostics | v5 | `encoderBlockFlow` корректно описывает обе execution orders. |
| Training Telemetry | v4 | Без изменения. |
| Metrics/OpenSearch | v10 | Без изменения. |

Semantic v5 является clean cut для новой runtime-границы. Она не принимает
Semantic v4 documents или v4 generations через Flight v22. Controlled A/B
повторно обучает и post-norm, и pre-norm generation на том же frozen artifact;
старые models не мигрируются и не получают synthetic diagnostics.

Новый language revision меняет D1 preimages через
`objectiveLanguageRevision: 5`. `dataContractSha256`, physical Arrow schemas,
`indexedFeatureBlocks`, objective formulas, target ordering, optimizer, AMP,
gradient clipping и training telemetry остаются неизменными. Не вводится
representation-independent hashing.

## Validation и outcomes

Semantic schema отклоняет отсутствие или неизвестное
`encoderNormalizationOrder` как `INVALID_ARGUMENT / INVALID_MODEL_CONTRACT`
в `modelTuning`. Это не отдельная предметная ошибка: нарушен closed model
contract.

Target Head Diagnostics v5 сохраняет существующие owner scope, availability,
cursor и integrity outcomes. Если stored artifact заявляет unknown order,
несогласованный порядок layers либо значения boundaries вне соответствующей
execution order, report возвращает существующий structured integrity outcome,
а не частичную проекцию.

## Контролируемая проверка после реализации

Inventory создаёт две новые L3 generation с разными exact ModelContract:

```text
baseline:    encoderNormalizationOrder = postNorm
experiment:  encoderNormalizationOrder = preNorm
```

Сравниваются `modelDefinitionSha256`, loss/health из Training Telemetry,
`representationFlow`, `encoderBlockFlow`, target-head raw logits и public
predictions из Diagnostics v5. Результат pre-norm не интерпретируется как
обобщающая способность или доказательство дефекта PyTorch; он отвечает только
на вопрос о роли order normalization в воспроизводимом L3 scenario.

## Граница текущей работы

Подготовлен staged canonical package со schemas, error outcomes, fixtures и
fixture manifests для перечисленных revision. До точного cross-project review
не меняются runtime, migrations, active contracts или deployment.
