"""Scaffolded sub-question chains (SCOPE-RL Stage 1 / Fig 1a probe).

The paper's scaffold prompt instructs the model to emit one \\boxed{} answer per
sub-problem, labeled `[SUB-X ANSWER]`, and `[MAIN ANSWER]` for the main problem.
Sub-answers are verifier-only: they live in `data/dapo_math_2400_scaffold.jsonl`
and are never shown to the policy (answer-hidden non-leakage, Eq. 6 predicate H).
"""

import json
import re

from verifier import last_boxed_only_string, normalize_final_answer, remove_boxed

LABEL_RE = re.compile(r"\[\s*(?:SUB[-\s]?(\d+)|(MAIN))\s+ANSWER\s*\]", re.IGNORECASE)


def load_scaffolds(path):
    """id -> {messages, sub_gts (ordered list), ground_truth}."""
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _first_boxed(segment: str):
    """First \\boxed{...} content in the segment (balanced-brace scan)."""
    idx = segment.find("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    depth = 0
    for j in range(i, min(len(segment), i + 2000)):
        if segment[j] == "{":
            depth += 1
        elif segment[j] == "}":
            depth -= 1
            if depth == 0:
                return segment[i + 1 : j]
    return None


def extract_labeled_answers(text: str, n_subs: int):
    """Return ([sub1_pred, ..., subN_pred], main_pred); missing answers are None.

    Each prediction is the first \\boxed{} content following the (last occurrence
    of the) corresponding label, cut off at the next label.
    """
    marks = []  # (pos_end, key)
    for m in LABEL_RE.finditer(text):
        key = "main" if m.group(2) else int(m.group(1))
        marks.append((m.end(), m.start(), key))
    subs = [None] * n_subs
    main = None
    for i, (end, start, key) in enumerate(marks):
        seg_stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        pred = _first_boxed(text[end:seg_stop])
        if pred is None:
            continue
        if key == "main":
            main = pred
        elif isinstance(key, int) and 1 <= key <= n_subs:
            subs[key - 1] = pred  # later occurrences of a label overwrite earlier
    if main is None:  # fall back to the last box anywhere in the tail
        boxed = last_boxed_only_string(text[-1000:])
        if boxed is not None:
            try:
                main = remove_boxed(boxed)
            except AssertionError:
                main = None
    return subs, main


def answer_matches(pred, gt: str) -> bool:
    if pred is None:
        return False
    return normalize_final_answer(str(pred)) == normalize_final_answer(str(gt))


def prefix_len(sub_correct) -> int:
    """Length of the longest all-correct prefix (paper Eq. 9, Pi_i)."""
    n = 0
    for ok in sub_correct:
        if not ok:
            break
        n += 1
    return n


def asr_reward(sub_correct, main_correct: bool, beta: float) -> float:
    """Prefix-consistent scaffold reward (paper Eq. 10).

    R = beta * (sum_i Pi_i)/m + (1-beta) * Pi_m * 1[main correct],
    where Pi_i = 1 iff subs 1..i are all correct, so sum_i Pi_i = prefix_len.
    """
    m = len(sub_correct)
    if m == 0:
        return float(main_correct)
    L = prefix_len(sub_correct)
    return beta * (L / m) + (1.0 - beta) * float(L == m) * float(main_correct)
