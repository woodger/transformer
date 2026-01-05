import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def save_model(model_name: str, model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    full_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model.state_dict(), full_path)

def load_model(model_name: str, model, device):
    full_path = os.path.join(MODELS_DIR, model_name)
    model.load_state_dict(torch.load(full_path, map_location=device))
    return model



def tree_stats(params):
    cpu_tensors = [p.detach().cpu().flatten() for p in params]
    flat = torch.cat(cpu_tensors)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "norm": float(torch.norm(flat)),
    }
