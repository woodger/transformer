# Байесовское ядро: Gaussian NLL

> Этот файл объясняет идеи и содержит исторические примеры. Исполняемый
> контракт текущих четырёх loss stages и точные формулы описаны в
> [`losses.md`](./losses.md), а публичная target-aligned семантика — в
> [`ADR 0007`](./adr/0007-target-aligned-flight-v4.md). Snippets ниже не
> являются актуальной архитектурой модели или CLI-схемой: упоминаемый в них
> `sigmaR` соответствует текущей private `returnScale`, а публичный
> `SigmaReturn` обучается отдельной прямой регрессией.

```
meanR     — ожидаемая доходность
sigmaR    — неопределённость
pTP, pSL  — вероятности исходов
volNext   — будущая волатильность
```

Для `meanR` и `sigmaR` — классика:

[
\mathcal{L}_{NLL}
= \frac{(r - \mu)^2}{2\sigma^2} + \log \sigma
]

### Почему лучше, чем MSE

* MSE **игнорирует уверенность**
* NLL **наказывает за самоуверенность**
* модель *учится говорить «я не уверен»*


### Код

```py
def gaussian_nll(pred_mean, pred_sigma, target):
    var = pred_sigma ** 2
    return torch.mean(
        (target - pred_mean) ** 2 / (2 * var) + torch.log(pred_sigma)
    )
```


# Байесовский EV (через Bernoulli)

TP / SL — это **дискретные события**
→ Bernoulli likelihood

Но **EV — это ожидание**:

[
EV = p_{TP} \cdot TP - p_{SL} \cdot SL
]

В Bayesian-версии мы:

* **не оптимизируем EV напрямую**
* а максимизируем его *апостериорное ожидание*
* * штрафуем за неопределённость


### Bayesian EV loss

```py
def bayesian_ev_loss(pTP, pSL, sigmaR, tp=1.0, sl=1.0):
    ev = pTP * tp - pSL * sl

    # uncertainty-aware penalty
    risk_penalty = sigmaR.detach() * torch.abs(ev)

    return -torch.mean(ev - 0.1 * risk_penalty)
```

🔑 `detach()` — важно:
мы **не даём sigma "читерить"**, снижая EV.


# Вероятности как распределения (очень важно)

Вместо:

```py
(pTP - t_pTP)^2
```

Используем **Binary Cross-Entropy**
(это и есть NLL Bernoulli)

```py
bce = torch.nn.functional.binary_cross_entropy
loss_prob = (
    bce(pTP, t_hitTP) +
    bce(pSL, 1 - t_hitTP)
)
```


# Волатильность = LogNormal

Формально:

[
\log(vol) \sim \mathcal{N}
]

```py
def lognormal_nll(pred_vol, target_vol):
    return torch.mean(
        (torch.log(pred_vol) - torch.log(target_vol)) ** 2
    )
```


# Финальный Bayesian combined_loss

Вот **готовая версия**, которую можно сразу использовать.

```py
def combined_loss(preds, targets, epoch, per_week):
    week = epoch // per_week + 1

    meanR, sigmaR, pTP, pSL, volNext, hitTP = preds.T
    t_meanR, _, _, _, t_volNext, t_hitTP = targets.T

    loss = 0.0

    # ----------------------    # 1. Gaussian NLL (return)
    # ----------------------    loss_ret = gaussian_nll(meanR, sigmaR, t_meanR)
    loss += loss_ret

    # ----------------------    # 2. Probability likelihood
    # ----------------------    if week >= 2:
        loss_prob = (
            torch.nn.functional.binary_cross_entropy(pTP, t_hitTP) +
            torch.nn.functional.binary_cross_entropy(pSL, 1 - t_hitTP)
        )
        loss += 0.5 * loss_prob

    # ----------------------    # 3. Bayesian EV
    # ----------------------    if week >= 3:
        loss_ev = bayesian_ev_loss(pTP, pSL, sigmaR)
        loss += 0.3 * loss_ev

    # ----------------------    # 4. Volatility (log-normal)
    # ----------------------    if week >= 4:
        loss_vol = lognormal_nll(volNext, t_volNext)
        loss += 0.2 * loss_vol

    return loss
```

# Концептуально что можно сделать дальше

1. **Kelly loss**
2. **Posterior Sharpe**
3. **Regime-aware sigma**
4. **Aleatoric vs Epistemic uncertainty**
