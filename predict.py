# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import glob
from typing import Optional
from pathlib import Path

from train_gpt2 import GPT, GPTConfig

import torch
from cog import BasePredictor, Input
import tiktoken


WEIGHTS = Path(__file__).parent / "weights" / "model.pth"

def _find_checkpoint() -> Optional[str]:
    if WEIGHTS.exists():
        return str(WEIGHTS)     # prefer weights/model.pth
    if Path("model.pth").exists():
        return "model.pth"      # fallback if you didn’t move it yet
    cks = sorted(Path("log").glob("model_*.pt"))
    return str(cks[-1]) if cks else None


def _maybe_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    cutoff = v[:, [-1]]
    logits = logits.clone()
    logits[logits < cutoff] = float("-inf")
    return logits


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load model and tokenizer once."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = tiktoken.get_encoding("gpt2")

        # Default config matches your training run
        cfg = GPTConfig(
            block_size=1024,
            vocab_size=50304,  # you used 50304 in training
            n_layer=12,
            n_head=12,
            n_embd=768,
        )

        ckpt_path = _find_checkpoint()
        if ckpt_path is None:
            # Cold start: random weights to keep the container usable
            self.model = GPT(cfg).to(self.device).eval()
            return

        obj = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj


        self.model = GPT(cfg).to(self.device)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # No prints: Cog logs stdout. Silent even if some buffers are missing.
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Prompt text to complete"),
        max_new_tokens: int = Input(description="Number of tokens to generate", ge=1, le=1024, default=128),
        temperature: float = Input(description="Softmax temperature", ge=0.05, le=5.0, default=1.0),
        top_k: int = Input(description="Top-k sampling. 0 disables", ge=0, le=1000, default=50),
        seed: int = Input(description="Random seed. Use -1 for random", default=0),
        stop_at_eos: bool = Input(description="Stop when <|endoftext|> (50256) appears", default=True),
        return_full_text: bool = Input(description="Return prompt+completion", default=False),
    ) -> str:
        """Generate a completion from the loaded model."""
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        # Encode prompt
        toks = self.enc.encode(prompt)
        x = torch.tensor(toks, dtype=torch.long, device=self.device)[None, :]
        eos_id = 50256  # GPT-2 EOT in tiktoken

        # Trim to block size
        bs = self.model.config.block_size
        if x.size(1) > bs:
            x = x[:, -bs:]

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward
                if x.size(1) > bs:
                    x = x[:, -bs:]
                logits, _ = self.model(x)
                logits = logits[:, -1, :] / temperature
                logits = _maybe_top_k(logits, top_k)
                probs = torch.softmax(logits, dim=-1)

                # Sample one token
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)

                if stop_at_eos and int(next_token.item()) == eos_id:
                    break

        out_ids = x[0].tolist()
        if return_full_text:
            return self.enc.decode(out_ids)
        else:
            gen_ids = out_ids[len(toks):]
            return self.enc.decode(gen_ids if gen_ids else [])
