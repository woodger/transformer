import torch


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path, model, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def tree_stats(params):
    cpu_tensors = [p.detach().cpu().flatten() for p in params]
    flat = torch.cat(cpu_tensors)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "norm": float(torch.norm(flat)),
    }
