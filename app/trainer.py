import torch
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext

from losses import combined_loss
from utils import save_model, load_model, tree_stats
from config import WEIGHT_DECAY, GRAD_CLIP_NORM


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

        best_loss = float("inf")
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for xb_cpu, yb_cpu in loader:
                xb = xb_cpu.to(self.device)
                yb = yb_cpu.to(self.device)

                self.optimizer.zero_grad()

                with self._autocast():
                    preds = self.model(xb)
                    loss = combined_loss(preds, yb, epoch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), GRAD_CLIP_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            epoch_loss = total_loss / len(loader)
            stats = tree_stats(self.model.parameters())

            print(
                f"epoch {epoch + 1}, "
                f"loss {epoch_loss:.6f}, "
                f"norm {stats['norm']:.0f}"
            )

            # if epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     wait = 0
            # else:
            #     wait += 1
            #     if wait >= self.patience:
            #         print("Early stopping")
            #         break

        save_model(model_name, self.model)
        print("Model saved")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), self._autocast():
            return self.model(X.to(self.device))

    def load(self, model_name: str):
        load_model(model_name, self.model, self.device)
