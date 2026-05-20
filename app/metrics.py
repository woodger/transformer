from dataclasses import dataclass


@dataclass
class TrainMetrics:
    rows: int = 0
    batches: int = 0
    loss: float = 0.0
    loss_ret: float = 0.0
    loss_prob: float = 0.0
    loss_ev: float = 0.0
    loss_vol: float = 0.0
    grad_norm: float = 0.0
    nan_ratio: float = 0.0
    valid_token_ratio: float = 0.0
    elapsed_ms: float = 0.0
    lr: float = 0.0
    week: int = 0

    def update(
        self,
        rows: int,
        loss_parts: dict[str, float],
        grad_norm: float,
        nan_ratio: float,
        valid_token_ratio: float,
    ):
        total_rows = self.rows + rows
        if total_rows <= 0:
            return

        def avg(current: float, value: float) -> float:
            return ((current * self.rows) + (value * rows)) / total_rows

        self.loss = avg(self.loss, loss_parts["loss"])
        self.loss_ret = avg(self.loss_ret, loss_parts["loss_ret"])
        self.loss_prob = avg(self.loss_prob, loss_parts["loss_prob"])
        self.loss_ev = avg(self.loss_ev, loss_parts["loss_ev"])
        self.loss_vol = avg(self.loss_vol, loss_parts["loss_vol"])
        self.grad_norm = avg(self.grad_norm, grad_norm)
        self.nan_ratio = avg(self.nan_ratio, nan_ratio)
        self.valid_token_ratio = avg(self.valid_token_ratio, valid_token_ratio)
        self.rows = total_rows
        self.batches += 1

    def log_line(self, **extra) -> str:
        fields = {
            **extra,
            "loss": f"{self.loss:.6f}",
            "ret": f"{self.loss_ret:.6f}",
            "prob": f"{self.loss_prob:.6f}",
            "ev": f"{self.loss_ev:.6f}",
            "vol": f"{self.loss_vol:.6f}",
            "grad": f"{self.grad_norm:.3f}",
            "rows": self.rows,
            "batches": self.batches,
            "nan": f"{self.nan_ratio:.4f}",
            "valid_tokens": f"{self.valid_token_ratio:.4f}",
            "lr": f"{self.lr:.6g}",
            "week": self.week,
            "ms": f"{self.elapsed_ms:.0f}",
        }

        return " ".join(f"{key}={value}" for key, value in fields.items())

    def __float__(self) -> float:
        return self.loss

    def __format__(self, format_spec: str) -> str:
        return format(self.loss, format_spec)

    def __gt__(self, other) -> bool:
        return self.loss > float(other)
