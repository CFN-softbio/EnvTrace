from envtrace.core.event import Event
from envtrace.core.alignment import DifflibAligner

def test_alignment_equal_sequences():
    gt = [Event("a", 0.0, 1, {}), Event("b", 0.1, 2, {}), Event("c", 0.2, 3, {})]
    pr = [Event("a", 0.0, 1, {}), Event("b", 0.1, 2, {}), Event("c", 0.2, 3, {})]
    aligned = DifflibAligner.align(gt, pr, key_fn=lambda e: e.channel)
    assert len(aligned) == 3
    for (g, p), ch in zip(aligned, ["a", "b", "c"]):
        assert g is not None and p is not None
        assert g.channel == ch and p.channel == ch

def test_alignment_insert_delete():
    gt = [Event("a", 0.0, 1, {}), Event("b", 0.1, 2, {}), Event("c", 0.2, 3, {})]
    pr = [Event("a", 0.0, 1, {}), Event("c", 0.2, 3, {})]  # 'b' missing
    aligned = DifflibAligner.align(gt, pr, key_fn=lambda e: e.channel)
    # Ensure one position aligned to None due to delete
    none_count = sum(1 for g, p in aligned if p is None or g is None)
    assert none_count >= 1
