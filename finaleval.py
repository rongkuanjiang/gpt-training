import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self.load_weights)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
            )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def _find_checkpoint(user_path: Optional[str]) -> Path:
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} does not exist")
        return path

    candidates = [
        Path(__file__).parent / "weights" / "model.pth",
        Path("model.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    log_dir = Path("log")
    ckpts = sorted(log_dir.glob("model_*.pt"))
    if ckpts:
        return ckpts[-1].resolve()
    raise FileNotFoundError(
        "Could not find a checkpoint. Pass --checkpoint or place model.pth in ./weights/."
    )


def _maybe_dataclass_config(cfg_obj: object) -> Optional[GPTConfig]:
    if isinstance(cfg_obj, GPTConfig):
        return cfg_obj
    if isinstance(cfg_obj, dict):
        return GPTConfig(**cfg_obj)
    return None


def _load_state_and_config(path: Path, device: torch.device) -> Tuple[dict, Optional[GPTConfig]]:
    raw_obj = torch.load(str(path), map_location=device)
    config = None
    state_dict = raw_obj

    if isinstance(raw_obj, dict) and "model" in raw_obj:
        state_dict = raw_obj["model"]
        config = _maybe_dataclass_config(raw_obj.get("config"))
    return state_dict, config


def _apply_overrides(cfg: GPTConfig, args: argparse.Namespace) -> GPTConfig:
    updates = {}
    for field in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    return replace(cfg, **updates) if updates else cfg


def load_model(checkpoint: Path, device: str, args: argparse.Namespace) -> GPT:
    device_obj = torch.device(device)
    state_dict, maybe_cfg = _load_state_and_config(checkpoint, device_obj)
    config = maybe_cfg or GPTConfig()
    config = _apply_overrides(config, args)
    model = GPT(config).to(device_obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing tensors in checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"[warn] unexpected tensors in checkpoint: {sorted(unexpected)}")
    model.eval()
    return model


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


@torch.no_grad()
def evaluate_hellaswag(
    model: GPT,
    device: str,
    split: str = "val",
    limit: Optional[int] = None,
    report_every: int = 100,
) -> Tuple[float, int, int]:
    num_correct = 0
    num_total = 0
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_enabled = device_type == "cuda"
    autocast_context = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16) if amp_enabled else nullcontext()
    )

    for idx, example in enumerate(iterate_examples(split)):
        if limit is not None and idx >= limit:
            break

        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with autocast_context:
            logits, _ = model(tokens)
        pred = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct += int(pred == label)

        if report_every and num_total % report_every == 0:
            acc = num_correct / num_total
            print(f"[eval] {num_total} examples -> accuracy {acc:.4f}")

    acc = num_correct / num_total if num_total else 0.0
    return acc, num_correct, num_total


def _determine_device(preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model on HellaSwag.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt/.pth weights.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, e.g. cuda or cpu.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (for smoke tests).")
    parser.add_argument("--report-every", type=int, default=200, help="Print running accuracy every N examples (0 to disable).")
    parser.add_argument("--block-size", type=int, default=None, help="Override block size in the checkpoint config.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size in the checkpoint config.")
    parser.add_argument("--n-layer", type=int, default=None, help="Override number of transformer layers.")
    parser.add_argument("--n-head", type=int, default=None, help="Override number of attention heads.")
    parser.add_argument("--n-embd", type=int, default=None, help="Override embedding dimension.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _determine_device(args.device)
    checkpoint = _find_checkpoint(args.checkpoint)

    print(f"Using checkpoint: {checkpoint}")
    print(f"Running on device: {device}")

    torch.set_float32_matmul_precision("high")
    model = load_model(checkpoint, device, args)

    acc, correct, total = evaluate_hellaswag(
        model,
        device,
        split=args.split,
        limit=args.max_examples,
        report_every=args.report_every,
    )

    print(f"HellaSwag {args.split} accuracy: {correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()
