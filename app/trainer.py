import torch
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext

from losses import combined_loss
from utils import save_model, load_model, tree_stats
from config import WEIGHT_DECAY, GRAD_CLIP_NORM

EV_START_EPOCH = 5  # например, с 2-й недели (EPOCHS_PER_WEEK = 5)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lr: float,
        batch_size: int,
        epochs: int,
        patience: int,
        use_amp: bool = False,
        weight_decay: float = WEIGHT_DECAY
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # AMP включаем только если GPU и user просил
        self.use_amp = bool(use_amp and device.type == "cuda")

        # GradScaler для AMP
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if use_amp and device.type != "cuda":
            print("AMP requested but CUDA not available — disabled")

        print(f"AMP enabled: {self.use_amp}")

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        else:
            return nullcontext()


    def fit(self, X: torch.Tensor, Y: torch.Tensor, model_name: str):
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        best_metric = None
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_ev = 0.0
            total_mse = 0.0

            for xb_cpu, yb_cpu in loader:
                xb = xb_cpu.to(self.device)
                yb = yb_cpu.to(self.device)

                self.optimizer.zero_grad()

                with self._autocast():
                    preds = self.model(xb)
                    loss, ev = combined_loss(preds, yb, epoch, return_ev=True)
                    mse = torch.mean((preds - yb) ** 2)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), GRAD_CLIP_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_ev += ev.item()
                total_mse += mse.item()

            epoch_loss = total_loss / len(loader)
            epoch_ev = total_ev / len(loader)
            epoch_mse = total_mse / len(loader)

            stats = tree_stats(self.model.parameters())

            print(
                f"epoch {epoch + 1}, "
                f"loss {epoch_loss:.6f}, "
                f"mse {epoch_mse:.6f}, "
                f"ev {epoch_ev:.4f}, "
                f"norm {stats['norm']:.0f}"
            )

            # -------------------------
            # Early stopping logic
            # -------------------------
            if epoch < EV_START_EPOCH:
                metric = -epoch_mse   # минимизируем MSE
            else:
                metric = epoch_ev     # максимизируем EV

            if best_metric is None or metric > best_metric:
                best_metric = metric
                wait = 0
                save_model(model_name, self.model)
                print("Model saved")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stopping")
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            return self.model(X.to(self.device))

    def load(self, model_name: str):
        load_model(model_name, self.model, self.device)
