## No-trade как отдельная голова

> Это ненормативная проектная идея, а не текущая архитектура. Любая её будущая
> реализация обязана сохранять шесть target-aligned public coordinates и
> private uncertainty head из
> [`ADR 0007`](./adr/0007-target-aligned-flight-v4.md). Примеры ниже созданы до
> Flight v4 и не задают wire-, checkpoint- или objective-контракт.

**Что это:**
Ввести **явный “no-trade” сигнал**.

Например:

* вероятность сделки
* или логит `trade / no_trade`
* или правило: `EV < 0 → no trade`

В трейдинге:
* *лучший трейд — это иногда отсутствие трейда*

Без этого:

* модель **вынуждена что-то предсказывать**
* шум превращается в сделки
* комиссии убивают результат

💡 Это критично для реального PnL.

## Концепция

Мы добавляем:

* отдельную **NoTradeHead**
* она предсказывает `p_trade ∈ (0, 1)`
* финальный выход =
  **торговые сигналы × p_trade**

> Модель учится **когда не стоит торговать**, а не просто «слабый сигнал»


## Архитектура (схема)

```
Transformer encoder
        ↓
 shared representation
        ↓
 ┌───────────────┐
 │  Regime heads │ → trading signals
 └───────────────┘
        ↓
 ┌───────────────┐
 │ No-trade head │ → p_trade
 └───────────────┘
        ↓
 final = trading * p_trade
```


## No-trade head

```py
class NoTradeHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # p_trade ∈ (0,1)
```


## Regime-aware head (без изменений логики)

(оставляем твой `RegimeAwareHead` как есть)


## Объединённая торговая голова

```py
class TradingDecisionHead(nn.Module):
    def __init__(self, hidden_dim, n_regimes=3):
        super().__init__()

        self.regime_head = RegimeAwareHead(hidden_dim, n_regimes)
        self.notrade_head = NoTradeHead(hidden_dim)

    def forward(self, x):
        """
        returns:
        preds: (B, 6)
        p_trade: (B, 1)
        """

        preds = self.regime_head(x)   # trading outputs
        p_trade = self.notrade_head(x)

        # мягкое выключение торговли
        preds_scaled = preds.clone()
        preds_scaled[:, :4] *= p_trade  # meanR, sigmaR, pTP, pSL
        preds_scaled[:, 4] *= p_trade.squeeze(1)  # vol
        preds_scaled[:, 5] *= p_trade.squeeze(1)  # hit

        return preds_scaled, p_trade
```


## Правка TransformerModel

```py
self.head = TradingDecisionHead(hidden_dim, n_regimes=3)
```

В `forward`:

```py
preds, p_trade = self.head(last_valid)
return preds, p_trade
```


## 🔧 Правка loss (кратко)

Добавь:

```py
def combined_loss(preds, targets, epoch, p_trade=None):
    ...

    if p_trade is not None:
        # штраф за постоянную торговлю
        notrade_penalty = torch.mean(p_trade)
        loss += 0.05 * notrade_penalty
```

👉 Это **ключевая строка** — без неё модель будет всегда торговать.


## Что теперь умеет модель

✅ Осознанно не торговать
✅ Снижать риск в шуме
✅ Переживать боковики
✅ Не «выдумывать» EV
✅ Быть ближе к реальной стратегии


## Важно (очень)

* **No-trade ≠ zero signals**
* Это **отдельное решение**, обучаемое
* Именно так делают в:

  * prop-фирмах
  * HFT
  * institutional models


Ниже — **ОДИН файл**, **минимальный**, **рабочий**, без лишней магии.


## `transformer.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------# Positional Encoding (simple)
# --------------------------class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --------------------------# Trading head (signals only)
# --------------------------class TradingHead(nn.Module):
    """
    Outputs RAW logits where needed (for AMP safety)
    Order:
    [meanR, sigmaR, pTP_logit, pSL_logit, vol, hit_logit]
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.mean_head = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)
        self.ptp_head = nn.Linear(128, 1)   # logits
        self.psl_head = nn.Linear(128, 1)   # logits
        self.vol_head = nn.Linear(128, 1)
        self.hit_head = nn.Linear(128, 1)   # logits

    def forward(self, x):
        h = self.shared(x)

        meanR = torch.tanh(self.mean_head(h))          # [-1,1]
        sigmaR = F.softplus(self.sigma_head(h)) + 1e-6 # >0

        pTP_logit = self.ptp_head(h)
        pSL_logit = self.psl_head(h)

        vol = F.softplus(self.vol_head(h)) + 1e-6
        hit_logit = self.hit_head(h)

        return torch.cat(
            [meanR, sigmaR, pTP_logit, pSL_logit, vol, hit_logit],
            dim=1
        )


# --------------------------# No-trade head
# --------------------------class NoTradeHead(nn.Module):
    """
    Outputs trade_logit
    sigmoid(trade_logit) = probability to trade
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)  # LOGIT (not sigmoid)


# --------------------------# Transformer Model
# --------------------------class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layers,
        dropout,
        nhead=8,
    ):
        super().__init__()

        assert hidden_dim % nhead == 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, layers)

        self.trade_head = TradingHead(hidden_dim)
        self.notrade_head = NoTradeHead(hidden_dim)

    def forward(self, x):
        """
        x: (B, S, input_dim)
        returns:
          preds: (B, 6)
          trade_logit: (B, 1)
        """

        key_padding_mask = torch.isnan(x).any(dim=-1)
        x = torch.nan_to_num(x, nan=0.0)

        x = self.input_proj(x)
        x = self.pos(x)

        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # last valid token
        lengths = (~key_padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = torch.arange(x.size(0), device=x.device)
        last = enc[idx, lengths]

        preds = self.trade_head(last)
        trade_logit = self.notrade_head(last)

        return preds, trade_logit
```


## Как использовать в training loop

```py
preds, trade_logit = model(x)

# trade probability
p_trade = torch.sigmoid(trade_logit)

# мягко выключаем торговлю
preds = preds * p_trade
```


## В чём проблема сейчас

Но, ведь модель может просто выдавать TP=0, SL=0?

Тогда:

* EV ≈ 0
* штрафов почти нет
* модель “ничего не делает” → локальный минимум

👉 Ты абсолютно прав: **это скрытый no-trade**, замаскированный под нулевые вероятности.


## Ключевая идея (важно)

> **No-trade — это решение, а не отсутствие сигнала**

Если модель торгует:

* она **обязана**:

  * иметь `pTP + pSL > 0`
  * взять на себя риск
  * платить штрафы

Если модель **не хочет брать риск**:

* она должна **явно сказать**: *я не торгую*

👉 Значит:
**нужно запретить “прятаться” через `pTP = pSL = 0`**


## Правильная архитектура (без таргета!)

### Trade gate = переключатель режима

```text
p_trade → {0, 1}
```

* `p_trade ≈ 0` → рынок плохой → no-trade
* `p_trade ≈ 1` → рынок торгуемый → trade


### Жёсткое правило: если trade → probabilities обязаны жить

**Лосс-constraint:**

```py
min_activity = torch.relu(0.2 - (pTP + pSL))
loss += λ * p_trade * min_activity
```

Если модель решила торговать (`p_trade ≈ 1`)
она **не может** поставить `pTP = pSL = 0`


### EV считается ТОЛЬКО если trade

```py
ev = p_trade * (pTP - pSL)
loss += -mean(ev)
```

* no-trade **честно обнуляет EV**
* но **не даёт халтурить**


### Risk penalty тоже gated

```py
risk_pen = p_trade * relu(sigmaR - abs(meanR))
```


## Почему no-trade таргет не нужен

| Подход                | Проблема    |
| --------------------- | ----------- |
| no-trade label        | субъективен |
| rule-based no-trade   | ломается    |
| **implicit via loss** | ✅ устойчиво |

Модель **сама находит**, когда:

* EV не положителен
* риск не оправдан
* volatility не сходится

👉 и **выключает торговлю**


## Итоговая логика

> **Торговать — дорого**
> **Не торговать — бесплатно**
> **Халтурить — запрещено**
