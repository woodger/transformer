import torch
import torch.nn.functional as F

from config import PER_WEEK


def combined_loss(
    preds,
    targets,
    step,
    per_week: int = PER_WEEK,
    return_parts: bool = False,
):
    if per_week <= 0:
        raise ValueError("per_week must be a positive integer")

    week = step // per_week + 1

    meanR, sigmaR, logitTP, logitSL, volNext, logitHit = preds.T
    t_meanR, _, _, _, t_volNext, t_hitTP = targets.T

    loss = preds.new_tensor(0.0)
    loss_prob = preds.new_tensor(0.0)
    loss_ev = preds.new_tensor(0.0)
    loss_vol = preds.new_tensor(0.0)

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
        raw_loss_prob = (
            F.binary_cross_entropy_with_logits(logitTP, t_hitTP) +
            F.binary_cross_entropy_with_logits(logitSL, 1 - t_hitTP)
        )
        loss_prob = 0.5 * raw_loss_prob
        loss += loss_prob

    # -------------------------
    # Bayesian EV
    # -------------------------
    if week >= 3:
        pTP = torch.sigmoid(logitTP)
        pSL = torch.sigmoid(logitSL)

        ev = pTP - pSL
        risk_pen = sigmaR.detach() * torch.abs(ev)
        loss_ev = -0.3 * torch.mean(ev - 0.1 * risk_pen)
        loss += loss_ev

    # -------------------------
    # Volatility
    # -------------------------
    if week >= 4:
        raw_loss_vol = torch.mean(
            (torch.log(volNext + 1e-6) - torch.log(t_volNext + 1e-6)) ** 2
        )
        loss_vol = 0.2 * raw_loss_vol
        loss += loss_vol

    if not return_parts:
        return loss

    sigma_values = sigmaR.detach().float().reshape(-1)

    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_ret": float(loss_ret.detach().cpu()),
        "loss_prob": float(loss_prob.detach().cpu()),
        "loss_ev": float(loss_ev.detach().cpu()),
        "loss_vol": float(loss_vol.detach().cpu()),
        "sigma_min": float(torch.min(sigma_values).cpu()),
        "sigma_p05": float(torch.quantile(sigma_values, 0.05).cpu()),
        "sigma_mean": float(torch.mean(sigma_values).cpu()),
        "week": week,
    }
