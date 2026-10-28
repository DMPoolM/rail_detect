"""Metric definitions: hand-computable cases so the numbers mean what they claim."""

from __future__ import annotations

import torch

from utils.metrics import evaluate

NUM_GRID, NUM_ROWS, B = 10, 4, 2


class FixedModel(torch.nn.Module):
    """Returns logits that force a chosen per-row class."""

    def __init__(self, classes: torch.Tensor):
        super().__init__()
        self.classes = classes  # (B, 2, 4)

    def forward(self, x):
        logits = torch.full((*self.classes.shape, NUM_GRID + 1), -20.0)
        logits.scatter_(-1, self.classes.unsqueeze(-1), 20.0)
        return logits, None


class OneBatchLoader:
    def __init__(self, label: torch.Tensor):
        self.label = label

    def __len__(self):
        return 1

    def __iter__(self):
        yield torch.zeros(B, 3, 8, 8), self.label


def test_perfect_prediction_scores_one():
    label = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 1, 1, 1], [2, 2, 2, 2]]])
    metrics = evaluate(FixedModel(label), OneBatchLoader(label), NUM_GRID, torch.device("cpu"))
    assert metrics["presence_precision"] == 1.0
    assert metrics["presence_recall"] == 1.0
    assert metrics["presence_f1"] == 1.0
    assert metrics["row_acc"] == 1.0
    assert metrics["row_mae_cells"] == 0.0


def test_missed_rows_hurt_recall_and_strict_row_acc():
    label = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 1, 1, 1], [2, 2, 2, 2]]])
    pred = label.clone()
    pred[0, 0, 0] = NUM_GRID  # pretend "no rail" on a row that has one
    metrics = evaluate(FixedModel(pred), OneBatchLoader(label), NUM_GRID, torch.device("cpu"))

    n_rows = label.numel()
    assert metrics["presence_recall"] == (n_rows - 1) / n_rows
    assert metrics["presence_precision"] == 1.0  # nothing falsely claimed
    # strict accuracy counts the miss as an error; the detected-only one does not
    assert metrics["row_acc"] == (n_rows - 1) / n_rows
    assert metrics["row_acc_detected"] == 1.0


def test_false_positives_hurt_precision():
    label = torch.tensor([[[1, 2, NUM_GRID, NUM_GRID], [5, 6, NUM_GRID, NUM_GRID]]] * B)
    pred = label.clone()
    pred[:, :, 2] = 3  # claim a rail where there is none
    metrics = evaluate(FixedModel(pred), OneBatchLoader(label), NUM_GRID, torch.device("cpu"))

    gt = int((label < NUM_GRID).sum())
    pd = int((pred < NUM_GRID).sum())
    assert metrics["presence_precision"] == gt / pd
    assert metrics["presence_recall"] == 1.0


def test_localisation_tolerance_and_pixel_error():
    label = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1]]] * B)
    pred = label.clone()
    pred[:, :, 0] += 2  # exactly at tolerance -> still a hit
    pred[:, :, 1] += 3  # outside tolerance -> a miss
    metrics = evaluate(
        FixedModel(pred), OneBatchLoader(label), NUM_GRID, torch.device("cpu"), tol=2, input_w=100
    )
    # 2 rails x 4 rows x 2 batches = 16 rows; only the "+3" ones miss -> 12/16
    assert metrics["row_acc"] == 0.75
    assert metrics["row_mae_cells"] > 0
    # 100 px / 10 cells = 10 px per cell
    assert metrics["row_mae_px"] > 0


def test_evaluate_restores_training_mode():
    model = FixedModel(torch.zeros(B, 2, NUM_ROWS, dtype=torch.long))
    model.train()
    evaluate(model, OneBatchLoader(torch.zeros(B, 2, NUM_ROWS, dtype=torch.long)), NUM_GRID, "cpu")
    assert model.training is True
