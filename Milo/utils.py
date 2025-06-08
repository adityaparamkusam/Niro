import os, torch, math, json

def cosine_decay(step, total_steps, warmup_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))

def save_checkpoint(step, model, opt, scaler, cfg):
    os.makedirs(cfg.model_dir, exist_ok=True)
    path = os.path.join(cfg.model_dir, f"ckpt_{step:06d}.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg":   cfg.__dict__
    }, path)
    print(f"[checkpoint] saved to {path}")

def load_checkpoint(path, model, opt, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt  .load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]