## Regime-aware heads

Разные “режимы рынка” → **разные головы модели**.

Примеры режимов:

* тренд / флэт
* высокая / низкая волатильность
* новости / спокойный рынок

Архитектурно:

```
encoder
   ↓
shared features
   ↓
[head_trend]   [head_range]   [head_high_vol]
```

Или:

* soft routing (веса)
* gating network

**Зачем тебе:**
Одна и та же логика **не работает во всех режимах**.
Это резко снижает:

* переобучение
* ложные сигналы

Мы делаем:

1. **Общий encoder**
2. **Несколько торговых голов** — по режимам рынка
3. **Gating-сеть**, которая:

   * смотрит на состояние рынка
   * решает, **какой режим сейчас**
   * либо **смешивает головы**


## Вариант 1 (рекомендуемый): Soft-gating (смесь голов)

> Самый стабильный вариант для начала.

### Архитектура

```
Transformer Encoder
        ↓
   shared features
        ↓
  ┌───────────────┐
  │   Gating MLP  │ → weights (softmax)
  └───────────────┘
      ↓     ↓     ↓
   Head 1  Head 2  Head 3
      ↓     ↓     ↓
      └── weighted sum ──→ final prediction
```


## Код: Regime-aware Trading Head

### Regime head (одна голова = твоя текущая TradingHead)

```py
class RegimeHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.mean = nn.Linear(128, 1)
        self.sigma = nn.Linear(128, 1)
        self.ptp = nn.Linear(128, 1)
        self.psl = nn.Linear(128, 1)
        self.vol = nn.Linear(128, 1)
        self.hit = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)

        meanR = torch.tanh(self.mean(h))
        sigmaR = F.softplus(self.sigma(h)) + 1e-6

        logitTP = self.ptp(h)
        logitSL = self.psl(h)
        logitHit = self.hit(h)

        volNext = F.softplus(self.vol(h)) + 1e-6

        return torch.cat(
            [meanR, sigmaR, logitTP, logitSL, volNext, logitHit],
            dim=1
        )
```


### Gating network (определяет режим)

```py
class RegimeGate(nn.Module):
    def __init__(self, hidden_dim, n_regimes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_regimes)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
```


### Regime-aware head (объединение)

```py
class RegimeAwareHead(nn.Module):
    def __init__(self, hidden_dim, n_regimes=3):
        super().__init__()

        self.heads = nn.ModuleList(
            [RegimeHead(hidden_dim) for _ in range(n_regimes)]
        )

        self.gate = RegimeGate(hidden_dim, n_regimes)

    def forward(self, x):
        # x: (B, hidden_dim)

        weights = self.gate(x)  # (B, R)

        preds = torch.stack(
            [head(x) for head in self.heads],
            dim=1
        )  # (B, R, 6)

        weights = weights.unsqueeze(-1)  # (B, R, 1)

        return torch.sum(preds * weights, dim=1)
```


## Встраивание в твой Transformer

```py
self.head = RegimeAwareHead(hidden_dim, n_regimes=3)
```

Больше ничего менять **не нужно**.


## Какие режимы выбрать сначала

Для старта **3 режима — идеально**:

1. **Low volatility / range**
2. **Trending**
3. **High volatility / news**

Модель **сама научится**, что есть что — без разметки.


## Почему это стабильно

* нет hard-switch (не дергается)
* нет if/else в forward
* AMP-safe
* backprop идёт через все головы
* gating обучается end-to-end


## Что можно добавить позже (по желанию)

* entropy regularization (чтобы режимы не схлопывались)
* hard routing после прогрева
* отдельные losses по режимам
* режим “no-trade” как отдельную голову
