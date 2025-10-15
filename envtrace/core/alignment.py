from __future__ import annotations
from typing import Protocol, Callable, List, Optional, Tuple
from envtrace.core.event import Event
import difflib

Pair = Tuple[Optional[Event], Optional[Event]]

class SequenceAligner(Protocol):
    def align(self, gt: List[Event], pred: List[Event], key_fn: Callable[[Event], str]) -> List[Pair]: ...

class DifflibAligner:
    @staticmethod
    def align(gt: List[Event], pred: List[Event], key_fn: Callable[[Event], str]) -> List[Pair]:
        gt_keys = [key_fn(e) for e in gt]
        pred_keys = [key_fn(e) for e in pred]
        sm = difflib.SequenceMatcher(None, gt_keys, pred_keys, autojunk=False)
        aligned_gt: List[Optional[Event]] = []
        aligned_pr: List[Optional[Event]] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    aligned_gt.append(gt[i1 + k])
                    aligned_pr.append(pred[j1 + k])
            elif tag == "delete":
                for k in range(i1, i2):
                    aligned_gt.append(gt[k])
                    aligned_pr.append(None)
            elif tag == "insert":
                for k in range(j1, j2):
                    aligned_gt.append(None)
                    aligned_pr.append(pred[k])
            else:  # replace
                span = max(i2 - i1, j2 - j1)
                for off in range(span):
                    aligned_gt.append(gt[i1 + off] if i1 + off < i2 else None)
                    aligned_pr.append(pred[j1 + off] if j1 + off < j2 else None)
        return list(zip(aligned_gt, aligned_pr))
