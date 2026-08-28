"""Training loop. Run: python -m yiwm.train  [--data eco|synth]"""

import argparse
import os

import torch

from .data import get_dataset
from .losses import yi_world_loss
from .model import YiWorldModel


def train(
    steps: int = 3000,
    batch_size: int = 256,
    lr: float = 2e-3,
    device: str = "cpu",
    ckpt: str = "checkpoints/yiwm.pt",
    data: str = "eco",
    log_every: int = 200,
    seed: int = 0,
    soft_hex_temp: float = 0.0,
    yao_norm: bool = False,
    text_encoder: str = "hash",
    synth_pool: int = 0,
    semantic_data: str | None = None,
):
    torch.manual_seed(seed)
    if semantic_data:
        from .data import SemanticJsonlDataset

        make = SemanticJsonlDataset(semantic_data, text_encoder)
        obs_dim, data = make.obs_dim, "semantic"
    else:
        make, obs_dim = get_dataset(data, text_encoder, synth_pool)
    model = YiWorldModel(obs_dim=obs_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    norm = None
    if yao_norm:
        warm = torch.cat([make(1024, seed=10_000 + i)["yao_target"] for i in range(4)])
        norm = (warm.mean(0, keepdim=True), warm.std(0, keepdim=True).clamp(min=1e-6))
        print("yao_target channel std:", [round(x, 3) for x in norm[1].flatten().tolist()])

    for s in range(1, steps + 1):
        batch = make(batch_size, device=device)
        out = model(
            batch["obs"], batch["entity_states"],
            batch["entity_cats"], batch["entity_adj"],
        )
        L = yi_world_loss(out, batch, soft_hex_temp=soft_hex_temp, yao_norm=norm)
        opt.zero_grad()
        L["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if s == 1 or s % log_every == 0:
            with torch.no_grad():
                hx = (out["hex_logits"].argmax(1) == batch["hex"]).float().mean().item()
                nx = (out["hex_logits_next"].argmax(1) == batch["hex_next"]).float().mean().item()
                ac = (out["policy"]["action_logits"].argmax(1) == batch["action"]).float().mean().item()
                mv = ((out["change"] > 0.5).long() == batch["moving"]).float().mean().item()
            print(
                f"step {s:5d} | loss {L['total']:.3f} | "
                f"benGua {hx:.3f} | zhiGua {nx:.3f} | moving {mv:.3f} | action {ac:.3f}",
                flush=True,
            )

    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "data": data,
         "obs_dim": obs_dim, "text_encoder": text_encoder},
        ckpt,
    )
    print("saved", ckpt)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="checkpoints/yiwm.pt")
    ap.add_argument("--data", choices=["eco", "synth"], default="eco")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--soft-hex-temp", type=float, default=0.0,
                    help="0 = hard CE (default). >0 enables soft 本卦 target; try ~0.1 with --data synth")
    ap.add_argument("--yao-norm", action="store_true",
                    help="per-channel standardise yao_target before the yao regression (usually unnecessary; tanh already tames scale)")
    ap.add_argument("--text-encoder", choices=["hash", "minilm", "minilm-ml"], default="hash",
                    help="synth obs source: hash (offline, no synonym generalisation) or a FROZEN sentence-transformer")
    ap.add_argument("--synth-pool", type=int, default=0,
                    help="pre-generate+embed a fixed pool of this many synth rows (needed to train the slow ST encoders); 0 = fresh every batch")
    ap.add_argument("--semantic-data", default=None,
                    help="path to a JSONL from augment.build_semantic_jsonl; overrides --data")
    a = ap.parse_args()
    train(a.steps, a.batch_size, a.lr, a.device, a.ckpt, a.data, a.log_every,
          soft_hex_temp=a.soft_hex_temp, yao_norm=a.yao_norm, semantic_data=a.semantic_data,
          text_encoder=a.text_encoder, synth_pool=a.synth_pool)
