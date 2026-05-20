import torch
import torch.nn.functional as F

from config import PER_WEEK

def combined_loss(preds, targets, epoch, per_week: int = PER_WEEK):
    if per_week <= 0:
        raise ValueError("per_week must be a positive integer")

    week = epoch // per_week + 1

    meanR, sigmaR, logitTP, logitSL, volNext, logitHit = preds.T
    t_meanR, _, _, _, t_volNext, t_hitTP = targets.T

    loss = 0.0

    # -------------------------
    # Gaussian NLL
    # -------------------------
    var = sigmaR ** 2 + 1e-6
    loss_ret = torch.mean(
        (t_meanR - meanR) ** 2 / (2 * var) + torch.log(sigmaR)
    )
    loss += loss_ret

    # -------------------------
    # Probabilities (AMP safe)
    # -------------------------
    if week >= 2:
        loss_prob = (
            F.binary_cross_entropy_with_logits(logitTP, t_hitTP) +
            F.binary_cross_entropy_with_logits(logitSL, 1 - t_hitTP)
        )
        loss += 0.5 * loss_prob

    # -------------------------
    # Bayesian EV
    # -------------------------
    if week >= 3:
        pTP = torch.sigmoid(logitTP)
        pSL = torch.sigmoid(logitSL)

        ev = pTP - pSL
        risk_pen = sigmaR.detach() * torch.abs(ev)
        loss += -0.3 * torch.mean(ev - 0.1 * risk_pen)

    # -------------------------
    # Volatility
    # -------------------------
    if week >= 4:
        loss_vol = torch.mean(
            (torch.log(volNext + 1e-6) - torch.log(t_volNext + 1e-6)) ** 2
        )
        loss += 0.2 * loss_vol

    return loss
