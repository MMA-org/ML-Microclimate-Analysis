# File: tests/test_metrics.py
import os
import sys

from tests.mocks.data import mock_batch, mock_batch_with_ignore  # adjust if needed
from ucs.utils.metrics import SegMetrics, TestMetrics

# Add the project root to sys.path so that the tests package is recognized.


def test_seg_metrics_init():
    seg_metrics = SegMetrics(
        num_classes=2, prefix="test_seg", ignore_index=255, average="macro"
    )
    assert seg_metrics.num_classes == 2
    # Note: The prefix is set with a trailing slash in the constructor.
    assert seg_metrics.prefix == "test_seg/"
    assert seg_metrics.ignore_index == 255
    assert seg_metrics.average == "macro"


def test_seg_metrics_forward(mock_batch):
    preds, targets = mock_batch
    seg_metrics = SegMetrics(
        num_classes=2, prefix="test_seg", ignore_index=255, average="macro"
    )
    out = seg_metrics.forward(preds, targets)
    # Check that the output is a dictionary with the expected metric keys.
    assert isinstance(out, dict), "Output should be a dictionary of metrics."
    assert "test_seg/mean_iou" in out
    assert "test_seg/mean_dice" in out


def test_seg_metrics_update_compute(mock_batch):
    preds, targets = mock_batch
    seg_metrics = SegMetrics(
        num_classes=2, prefix="test_seg", ignore_index=255, average="macro"
    )
    seg_metrics.update(preds, targets)
    results = seg_metrics.compute()
    # Check expected keys and that metric values are between 0 and 1.
    assert "test_seg/mean_iou" in results
    assert "test_seg/mean_dice" in results
    mean_iou = results["test_seg/mean_iou"]
    mean_dice = results["test_seg/mean_dice"]
    assert 0.0 <= mean_iou <= 1.0
    assert 0.0 <= mean_dice <= 1.0


def test_test_metrics_init():
    test_metrics = TestMetrics(
        num_classes=2, prefix="test_eval", ignore_index=255, average="weighted"
    )
    assert test_metrics.num_classes == 2
    assert test_metrics.prefix == "test_eval/"
    assert test_metrics.ignore_index == 255
    assert test_metrics.average == "weighted"


def test_test_metrics_forward(mock_batch):
    preds, targets = mock_batch
    test_metrics = TestMetrics(
        num_classes=2, prefix="test_eval", ignore_index=255, average="weighted"
    )
    out = test_metrics.forward(preds, targets)
    # Check that all expected metric keys are present.
    assert isinstance(out, dict), "Output should be a dictionary of metrics."
    for key in [
        "test_eval/mean_iou",
        "test_eval/mean_dice",
        "test_eval/accuracy",
        "test_eval/precision",
        "test_eval/recall",
    ]:
        assert key in out, f"Missing {key} in output."


def test_test_metrics_update_compute(mock_batch):
    preds, targets = mock_batch
    test_metrics = TestMetrics(
        num_classes=2, prefix="test_eval", ignore_index=255, average="weighted"
    )
    test_metrics.update(preds, targets)
    results = test_metrics.compute()
    # Check that all expected keys are present.
    for key in [
        "test_eval/mean_iou",
        "test_eval/mean_dice",
        "test_eval/accuracy",
        "test_eval/precision",
        "test_eval/recall",
    ]:
        assert key in results, f"Missing {key} in results."
        val = results[key]
        assert 0.0 <= val <= 1.0, f"Metric {key} is out of range: {val}"


def test_seg_metrics_forward_ignore(mock_batch_with_ignore):
    """
    Test the forward method of SegMetrics with masks that include ignore index.
    """
    preds, targets = mock_batch_with_ignore
    seg_metrics = SegMetrics(
        num_classes=2, prefix="test_seg", ignore_index=255, average="macro"
    )
    assert (targets == 255).any(), "Expected ignore index 255 in targets"
    out = seg_metrics.forward(preds, targets)
    # Check that the output is a dictionary with the expected metric keys.
    assert isinstance(out, dict), "Output should be a dictionary of metrics."
    assert "test_seg/mean_iou" in out
    assert "test_seg/mean_dice" in out


def test_seg_metrics_update_compute_ignore(mock_batch_with_ignore):
    """
    Test update and compute methods of SegMetrics with ignore index in masks.
    """
    preds, targets = mock_batch_with_ignore
    assert (targets == 255).any(), "Expected ignore index 255 in targets"
    seg_metrics = SegMetrics(
        num_classes=2, prefix="test_seg", ignore_index=255, average="macro"
    )
    seg_metrics.update(preds, targets)
    results = seg_metrics.compute()
    # Check that expected keys are present and values are within [0, 1].
    assert "test_seg/mean_iou" in results
    assert "test_seg/mean_dice" in results
    for key in ["test_seg/mean_iou", "test_seg/mean_dice"]:
        val = results[key]
        assert 0.0 <= val <= 1.0, f"Metric {key} is out of range: {val}"


def test_test_metrics_forward_ignore(mock_batch_with_ignore):
    """
    Test the forward method of TestMetrics with masks that include ignore index.
    """
    preds, targets = mock_batch_with_ignore
    assert (targets == 255).any(), "Expected ignore index 255 in targets"

    test_metrics = TestMetrics(
        num_classes=2, prefix="test_eval", ignore_index=255, average="weighted"
    )
    out = test_metrics.forward(preds, targets)
    assert isinstance(out, dict), "Output should be a dictionary of metrics."
    expected_keys = [
        "test_eval/mean_iou",
        "test_eval/mean_dice",
        "test_eval/accuracy",
        "test_eval/precision",
        "test_eval/recall",
    ]
    for key in expected_keys:
        assert key in out, f"Missing {key} in output."


def test_test_metrics_update_compute_ignore(mock_batch_with_ignore):
    """
    Test update and compute methods of TestMetrics with ignore index in masks.
    """
    preds, targets = mock_batch_with_ignore
    assert (targets == 255).any(), "Expected ignore index 255 in targets"

    test_metrics = TestMetrics(
        num_classes=2, prefix="test_eval", ignore_index=255, average="weighted"
    )
    test_metrics.update(preds, targets)
    results = test_metrics.compute()
    expected_keys = [
        "test_eval/mean_iou",
        "test_eval/mean_dice",
        "test_eval/accuracy",
        "test_eval/precision",
        "test_eval/recall",
    ]
    for key in expected_keys:
        assert key in results, f"Missing {key} in results."
        val = results[key]
        assert 0.0 <= val <= 1.0, f"Metric {key} is out of range: {val}"
