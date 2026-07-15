"""Rule-based answer verification.

Math checking is ported verbatim from the paper repo's
verl/recipe/scope_rl/reward_score/math_dapo.py (Apache-2.0, adapted from
lm-evaluation-harness Minerva math utils), so train-time rewards and eval-time
accuracy match the paper's verifier. GPQA rows (multiple choice A-D) are scored
by extracting the chosen letter.
"""

import re
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> tuple[bool, str]:
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)
    return (pred == gt), pred


def verify_math(solution_str: str, answer: str) -> tuple[bool, str]:
    """Paper's compute_score: look at the tail of the response, try the
    'Answer: ...' pattern; fall back to the last \\boxed{} if no Answer line."""
    tail = solution_str[-300:]
    correct, pred = is_correct_minerva(tail, answer)
    if not correct and pred == "[INVALID]":
        boxed = last_boxed_only_string(solution_str[-1000:])
        if boxed is not None:
            pred = normalize_final_answer(remove_boxed(boxed))
            correct = pred == normalize_final_answer(answer)
    return correct, pred


def verify_choice(solution_str: str, answer: str) -> tuple[bool, str]:
    """Multiple-choice (GPQA): extract the chosen letter from the tail."""
    tail = solution_str[-300:]
    gt = answer.strip().strip(".")[:1].upper()
    m = re.findall(r"(?i)Answer\s*:\s*\**\(?([A-D])\)?", tail)
    if not m:
        boxed = last_boxed_only_string(tail)
        if boxed is not None:
            inner = remove_boxed(boxed).strip()
            m = re.findall(r"\(?([A-D])\)?", inner[:3])
    if not m:
        m = re.findall(r"(?i)\b(?:option|choice)\s*\(?([A-D])\)?", tail)
    pred = m[-1].upper() if m else "[INVALID]"
    return pred == gt, pred


def score_response(solution_str: str, ground_truth: str, source: str = "math") -> float:
    if source.startswith("gpqa"):
        correct, _ = verify_choice(solution_str, ground_truth)
    else:
        correct, _ = verify_math(solution_str, ground_truth)
    return 1.0 if correct else 0.0
