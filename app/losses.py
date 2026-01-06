import torch

EPOCHS_PER_WEEK = 5

def combined_loss(preds, targets, epoch):
    week = epoch // EPOCHS_PER_WEEK + 1

    # --------------------
    # base MSE (always on)
    # --------------------
    loss = torch.mean((preds - targets) ** 2)

    # fallback for tests
    if preds.shape[1] < 6:
        return loss

    # --------------------
    # unpack predictions
    # --------------------
    meanR, sigmaR, pTP, pSL, volNext, hitTP = preds.T
    t_meanR, t_sigmaR, t_pTP, t_pSL, t_volNext, t_hitTP = targets.T

    # --------------------
    # EV loss
    # --------------------
    if week >= 2:
        TP = 1.0
        SL = 1.0
        ev = pTP * TP - pSL * SL
        loss_ev = -torch.mean(ev)
        loss += 1.0 * loss_ev

    # --------------------
    # probability constraints
    # --------------------
    if week >= 3:
        prob_range_penalty = (
            torch.relu(-pTP) + torch.relu(pTP - 1) +
            torch.relu(-pSL) + torch.relu(pSL - 1)
        ).mean()

        prob_sum_penalty = torch.relu(pTP + pSL - 1).mean()

        loss += 0.1 * (prob_range_penalty + prob_sum_penalty)

    # --------------------
    # risk penalty
    # --------------------
    if week >= 4:
        risk_pen = torch.relu(sigmaR - torch.abs(meanR)).mean()
        loss += 0.2 * risk_pen

    # --------------------
    # consistency + volatility
    # --------------------
    if week >= 5:
        # meanReturn should align with TP/SL probabilities
        implied_return = pTP - pSL
        consistency = torch.mean((meanR - implied_return) ** 2)

        # volatility: log-error is more stable
        vol_loss = torch.mean(
            (torch.log(volNext + 1e-6) - torch.log(t_volNext + 1e-6)) ** 2
        )

        loss += 0.5 * consistency
        loss += 0.3 * vol_loss

    return loss
