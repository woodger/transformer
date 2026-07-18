# Функция потерь

Текущая модель всегда возвращает шесть значений на строку:

| Позиция | Выход модели | Семантика |
| --- | --- | --- |
| `0` | `meanR` | ожидаемый return, `tanh`, диапазон `[-1, 1]` |
| `1` | `sigmaR` | положительный масштаб return distribution |
| `2` | `logitTP` | логит вероятности take-profit |
| `3` | `logitSL` | логит вероятности stop-loss |
| `4` | `volNext` | положительный прогноз следующей volatility |
| `5` | `logitHit` | дополнительный логит; текущий loss его не использует |

Training target также имеет ширину ровно `6`. Текущая функция потерь использует
только следующие позиции:

- `tgt[:, 0]` — target return;
- `tgt[:, 4]` — target volatility;
- `tgt[:, 5]` — hit probability в диапазоне `[0, 1]`.

Позиции `tgt[:, 1:4]` принимаются как часть формата, но текущим loss не
используются.

## Этапы

Loss накапливает компоненты по четырём этапам:

| Stage | Имя | Активные компоненты |
| --- | --- | --- |
| `1` | `returns` | `ret` |
| `2` | `probabilities` | `ret + prob` |
| `3` | `bayesian-ev` | `ret + prob + ev` |
| `4` | `volatility` | `ret + prob + ev + vol` |

### Stage 1: return

Используется Gaussian NLL с согласованной дисперсией:

```text
var = sigmaR² + 1e-6
loss_ret = mean(0.5 * ((target_meanR - meanR)² / var + log(var)))
```

Gaussian NLL может быть отрицательным при малой дисперсии; само по себе
отрицательное значение не означает ошибку вычисления.

### Stage 2: probabilities

К return-компоненту добавляются две BCE-with-logits цели:

```text
loss_prob = 0.5 * (
    BCE(logitTP, target_hit_probability)
    + BCE(logitSL, 1 - target_hit_probability)
)
```

### Stage 3: Bayesian EV

```text
pTP = sigmoid(logitTP)
pSL = sigmoid(logitSL)
ev = pTP - pSL
risk = detach(sigmaR) * abs(ev)
loss_ev = -0.3 * mean(ev - 0.1 * risk)
```

`detach(sigmaR)` означает, что EV-компонент не изменяет `sigmaR` своим
градиентом.

### Stage 4: volatility

```text
loss_vol = 0.2 * mean(
    (log(volNext + 1e-6) - log(target_volNext + 1e-6))²
)
```

## Schedule

`--loss-stage` задаёт максимальный stage, а `--loss-schedule` — способ перехода:

- `none` — сразу и постоянно используется `--loss-stage`;
- `epoch` — stage повышается каждые `--stage-size` эпох внутри текущего frame;
- `step` — stage повышается каждые `--stage-size` глобальных optimizer steps и
  может смениться посреди epoch.

При defaults `--loss-stage=4 --loss-schedule=epoch --stage-size=5` эпохи
`1..5` используют stage 1, `6..10` — stage 2, `11..15` — stage 3, а с эпохи
`16` используется stage 4. В `fit-stream` epoch schedule начинается заново для
каждого frame; step schedule продолжает глобальный счётчик между frames.
