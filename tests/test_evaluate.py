import pytest

from climbtrack import evaluate


def test_iou():
    assert evaluate.iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert evaluate.iou((0, 1), (2, 3)) == 0


def test_event_scores_one_to_one():
    gt = [(0, 10), (20, 30)]
    pred = [(1, 10), (2, 9), (50, 60)]  # duplicate detection must not double count
    e = evaluate.event_scores(gt, pred)
    assert e["tp"] == 1 and e["precision"] == pytest.approx(1 / 3) and e["recall"] == 0.5
    assert e["count_error"] == 1


def test_frame_scores_perfect():
    f = evaluate.frame_scores([(0, 10)], [(0, 10)], 20)
    assert f["frame_f1"] == pytest.approx(1.0)
