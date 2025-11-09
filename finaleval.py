import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from types import SimpleNamespace

from hellaswag import iterate_examples, render_example
from train_gpt2 import GPT, GPTConfig

WEIGHTS = Path(__file__).parent / "weights" / "model.pth"
MAX_EXAMPLES = None  # set to an int for a quicker smoke test


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



def main() -> None:
    # if not WEIGHTS.exists():
    #     raise FileNotFoundError(f"Missing checkpoint at {WEIGHTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    torch.set_float32_matmul_precision("high")
    model = GPT(GPTConfig()).to(device)
    # model = torch.compile(model)

    state = torch.load(str(WEIGHTS), map_location=device)
    # if isinstance(checkpoint, dict) and "model" in checkpoint:
    #     checkpoint = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {sorted(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {sorted(unexpected)}")
    model.eval()

    num_correct = 0
    num_total = 0
    for idx, example in enumerate(iterate_examples("val")):
        if MAX_EXAMPLES is not None and idx >= MAX_EXAMPLES:
            break
        
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits, _ = model(tokens)
        pred = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct += int(pred == label)
        if num_total % 500 == 0:
            print(f"{num_total} examples evaluated -> accuracy {num_correct / num_total:.4f}")

    print(f"HellaSwag val accuracy: {num_correct}/{num_total} = {num_correct / num_total:.4f}")


if __name__ == "__main__":
    main()
