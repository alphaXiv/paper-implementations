import bisect
from typing import Type

import regex as re
from tqdm import tqdm

from .aligners import AlignerRegistry, BaseAligner
from .registry import BaseRegistry
from .segmenters import BaseSegmenter, SegmenterRegistry


class TextMetricRegistry(BaseRegistry[Type["BaseTextMetric"]]):
    """A registry for text metrics."""


class BaseTextMetric:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute(self, gold: str, pred: str) -> float:
        raise NotImplementedError()

    def batch_compute(self, golds: list[str], preds: list[str]) -> list[float]:
        it = tqdm(
            zip(golds, preds),
            total=min(len(golds), len(preds)),
            desc=type(self).__name__,
            unit="samples",
            unit_scale=True,
        )
        return [self.compute(gold, pred) for gold, pred in it]


class BaseTextAlignMetric(BaseTextMetric):
    def __init__(
        self,
        segmenter: str | BaseSegmenter,
        aligner: str | BaseAligner = "hirschberg",
        aligner_kwargs: dict = {},
        segmenter_kwargs: dict = {},
        gap_token: str = "▓",
        *args,
        **kwargs,
    ):
        if isinstance(segmenter, str):
            self.segmenter = SegmenterRegistry.get(segmenter)(segmenter, **segmenter_kwargs)
        else:
            self.segmenter = segmenter

        if isinstance(aligner, str):
            self.aligner = AlignerRegistry.get(aligner)(aligner, **aligner_kwargs)
        else:
            self.aligner = aligner

        self.gap_token = gap_token

    def segment(self, seq_a_tokens: list[str], seq_b_tokens: list[str]) -> list[tuple[list[str], list[str]]]:
        return [(seq_a_tokens, seq_b_tokens)]

    def align(self, seq_a_tokens: list[str], seq_b_tokens: list[str]) -> tuple[list[str], list[str]]:
        return self.aligner.align(seq_a_tokens, seq_b_tokens)

    def tokenize(self, text: str) -> list[str]:
        return [w for w in re.split(r"(\p{P}+|\s+)", text) if w]

    def compute(self, gold: str, pred: str) -> float:
        raise NotImplementedError()


@TextMetricRegistry.add("document_edit_similarity")
class DocumentEditSimilarity(BaseTextAlignMetric):
    def _score_aligned(self, aligned_gold_tokens: list[str], aligned_pred_tokens: list[str]) -> float:
        insertions = deletions = matches = substitutions = 0.0
        for gold_symbol, pred_symbol in zip(aligned_gold_tokens, aligned_pred_tokens):
            if gold_symbol == self.gap_token:
                insertions += 1
            elif pred_symbol == self.gap_token:
                deletions += 1
            elif gold_symbol == pred_symbol:
                matches += 1
            else:
                substitutions += 1

        if total := insertions + deletions + matches + substitutions:
            return matches / total
        return 0.0

    def compute(self, gold: str, pred: str) -> float:
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)
        aligned_gold_tokens, aligned_pred_tokens = self.align(gold_tokens, pred_tokens)
        return self._score_aligned(aligned_gold_tokens, aligned_pred_tokens)


def find_align_gaps(aligned_text: list[str], gap_token: str = "▓", gap_threshold: int = 3) -> list[int]:
    consecutive_gaps_counter = 0
    above_threshold_locs: list[int] = []

    for aligned_pos, symbol in enumerate(aligned_text):
        if symbol == gap_token:
            consecutive_gaps_counter += 1
        else:
            consecutive_gaps_counter = 0

        if consecutive_gaps_counter >= gap_threshold:
            above_threshold_locs.append(aligned_pos)
            consecutive_gaps_counter = 0

    return above_threshold_locs


def make_unaligned_text(tokens: list[str], gap_token: str = "▓") -> str:
    return "".join(symbol for symbol in tokens if symbol != gap_token)


def find_sentences(
    tokens: list[str],
    sentences: list[str],
    gap_token: str = "▓",
):
    matches: list[tuple[int, int]] = []

    original_text = ""
    original: list[int] = []
    original_to_aligned: list[int] = []

    for i, token in enumerate(tokens):
        if token != gap_token:
            original_text += token
            original.append(len(original_text))
            original_to_aligned.append(i)

    matches = []
    for sentence in sentences:
        start_pos = original_text.find(sentence)
        if start_pos < 0:
            continue

        end_pos = start_pos + len(sentence)
        start_token = original_to_aligned[bisect.bisect_left(original, start_pos)]
        end_token = original_to_aligned[min(bisect.bisect_right(original, end_pos), len(original) - 1)]
        matches.append((start_token, end_token))

    return matches


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []

    # Sort spans based on start position
    sorted_spans = sorted(spans, key=lambda x: x[0])

    merged = [sorted_spans[0]]

    for current in sorted_spans[1:]:
        last = merged[-1]

        # If current span overlaps with last merged span, update the end of last span
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def make_sentences_around_gaps(sent_locs: list[tuple[int, int]], gaps_locs: list[int], window: int):
    sent_start_only = [start for start, _ in sent_locs]

    sentences_with_gaps = []

    # collect all sentences that are around the gaps
    for gap in gaps_locs:
        start_idx = bisect.bisect_left(sent_start_only, gap)
        fwd_window = max(0, start_idx - window)
        bwd_window = min(len(sent_locs) - 1, start_idx + window)
        sentences_with_gaps.append((sent_locs[fwd_window][0], sent_locs[bwd_window][-1]))

    # merge overlapping sentences
    sentences_with_gaps = merge_spans(sentences_with_gaps)

    return sentences_with_gaps


@TextMetricRegistry.add("paragraph_edit_similarity")
class ParagraphEditSimilarity(DocumentEditSimilarity):
    def __init__(
        self,
        segmenter: str | BaseSegmenter,
        aligner: str | BaseAligner = "hirschberg",
        aligner_kwargs: dict = {},
        segmenter_kwargs: dict = {},
        gap_token: str = "▓",
        gap_threshold: int = 3,
        sent_window: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            segmenter=segmenter,
            aligner=aligner,
            aligner_kwargs=aligner_kwargs,
            segmenter_kwargs=segmenter_kwargs,
            gap_token=gap_token,
        )
        self.gap_threshold = gap_threshold
        self.sent_window = sent_window

    def segment(self, seq_a_tokens: list[str], seq_b_tokens: list[str]) -> list[tuple[list[str], list[str]]]:
        all_spans = []

        for seq_tokens in (seq_a_tokens, seq_b_tokens):
            text = make_unaligned_text(tokens=seq_tokens, gap_token=self.gap_token)
            sentences = self.segmenter.segment(text)

            sent_locs = find_sentences(tokens=seq_tokens, sentences=sentences, gap_token=self.gap_token)
            gaps_locs = find_align_gaps(aligned_text=seq_tokens, gap_token=self.gap_token, gap_threshold=3)

            sentences_with_gaps = make_sentences_around_gaps(sent_locs=sent_locs, gaps_locs=gaps_locs, window=self.sent_window)
            all_spans.extend(sentences_with_gaps)

        return [(seq_a_tokens[start:end], seq_b_tokens[start:end]) for start, end in merge_spans(all_spans)]

    def compute(self, gold: str, pred: str) -> float:
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)
        aligned_gold_tokens, aligned_pred_tokens = self.align(gold_tokens, pred_tokens)

        scores = []
        for gold_segment, pred_segment in self.segment(aligned_gold_tokens, aligned_pred_tokens):
            score = self._score_aligned(gold_segment, pred_segment)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 1.0
