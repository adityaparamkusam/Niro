import os, time, torch
from dataclasses import asdict
from torch.amp import autocast, GradScaler           # backend-agnostic AMP

from .config  import ModelConfig, TrainConfig, save_config
from .dataset import create_loader
from .model   import GPT
from .utils   import cosine_decay, save_checkpoint


def main() -> None:
    print(" Starting Niro training …")
    # ── 1. instantiate & persist configs ──────────────────────────────────────
    m_cfg = ModelConfig()
    t_cfg = TrainConfig()
    os.makedirs(t_cfg.model_dir, exist_ok=True)
    save_config(
        {"model": asdict(m_cfg), "training": asdict(t_cfg)},
        os.path.join(t_cfg.model_dir, "config.json"),
    )

    # ── 2. data loader (add/remove paths as you like) ─────────────────────────
    data_paths = [
        "dataset/cc_news_tokenized",
        "dataset/tinystories_tokenized",
    ]
    train_loader = create_loader(data_paths, m_cfg, t_cfg)
    loader_iter  = iter(train_loader)

    # ── 3. model, optimiser, scaler ───────────────────────────────────────────
    device = torch.device(t_cfg.device)
    model  = GPT(m_cfg).to(device)
    print(f"Model params: {model.num_parameters():,.0f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg.lr, betas=(0.9, 0.95), weight_decay=0.1  # Added weight_decay
    )
    scaler = GradScaler()
    model.train()

    # ── 4. training loop ──────────────────────────────────────────────────────
    running_loss = grad_iter = 0.0
    for step in range(t_cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)

        # gradient accumulation
        for _ in range(t_cfg.grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:           # end of loader -> restart
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)

            x, y = x.to(device), y.to(device)
            with autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)

            scaler.scale(loss / t_cfg.grad_accum).backward()
            running_loss += loss.item()
            grad_iter    += 1

        # optimiser step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # LR schedule
        lr = cosine_decay(step, t_cfg.max_steps, t_cfg.warmup_steps, t_cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # logging
        if (step + 1) % t_cfg.log_every == 0:
            print(f"step {step+1:6d} | loss {running_loss/grad_iter:.4f} | lr {lr:.2e}")
            running_loss = grad_iter = 0.0

        # checkpoint
        if (step + 1) % t_cfg.ckpt_every == 0 or step + 1 == t_cfg.max_steps:
            save_checkpoint(step + 1, model, optimizer, scaler, t_cfg)


if __name__ == "__main__":
    main()