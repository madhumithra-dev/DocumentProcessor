"""
Standalone re-implementation of layoutreader.v3.helpers.
Drop this file next to document_processor.py — no pip install needed.
"""
import torch


def boxes2inputs(boxes: list[list[int]]) -> dict:
    """
    Convert normalised [0-1000] boxes to LayoutLMv3 inputs.
    Adds [CLS] at position 0 and [SEP] at the end.
    """
    n = len(boxes)

    # [CLS] token id = 0, [SEP] = 2  (LayoutLMv3 defaults)
    cls_id, sep_id, pad_box = 0, 2, [0, 0, 0, 0]

    input_ids = [cls_id] + [6] * n + [sep_id]          # 6 = generic token
    bbox      = [pad_box] + boxes + [pad_box]

    return {
        "input_ids":      torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, n + 2,       dtype=torch.long),
        "bbox":           torch.tensor([bbox],       dtype=torch.long),
    }


def prepare_inputs(inputs: dict, model) -> dict:
    """Move all tensors to the same device as the model."""
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in inputs.items()}


def parse_logits(logits: torch.Tensor, n_boxes: int) -> list[int]:
    """
    Greedy pointer-network decoding.
    logits shape: [seq_len, seq_len]
    Returns a list of length n_boxes where result[i] = reading position of box i.
    """
    # Only look at the n_boxes real tokens (skip [CLS] at 0)
    # logits[i] = distribution over which box to place at position i
    token_logits = logits[1 : n_boxes + 1, 1 : n_boxes + 1]  # [n, n]

    visited  = set()
    order    = [0] * n_boxes

    for position in range(n_boxes):
        scores = token_logits[position].clone()
        # mask already-assigned boxes
        for v in visited:
            scores[v] = -1e9
        chosen = int(scores.argmax().item())
        visited.add(chosen)
        order[chosen] = position   # box 'chosen' is at reading position 'position'

    return order
