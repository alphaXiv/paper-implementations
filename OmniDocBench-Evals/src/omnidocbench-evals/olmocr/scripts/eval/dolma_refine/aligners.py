from typing import Type

from sequence_align.pairwise import hirschberg, needleman_wunsch

from .registry import BaseRegistry


class AlignerRegistry(BaseRegistry[Type["BaseAligner"]]):
    """A registry for aligners."""


class BaseAligner:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def align(self, gold: list[str], pred: list[str]) -> tuple[list[str], list[str]]:
        raise NotImplementedError()


@AlignerRegistry.add("hirschberg")
class HirschbergAligner(BaseAligner):
    def __init__(
        self,
        match_score: float = 1.0,
        mismatch_score: float = -1.0,
        indel_score: float = -1.0,
        gap_token: str = "▓",
    ):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.indel_score = indel_score
        self.gap_token = gap_token
        super().__init__()

    def align(self, gold: list[str], pred: list[str]) -> tuple[list[str], list[str]]:
        return hirschberg(
            gold,
            pred,
            match_score=self.match_score,
            mismatch_score=self.mismatch_score,
            indel_score=self.indel_score,
            gap=self.gap_token,
        )


@AlignerRegistry.add("needleman-wunsch")
class NeedlemanWunschAligner(BaseAligner):
    def __init__(
        self,
        match_score: float = 1.0,
        mismatch_score: float = -1.0,
        indel_score: float = -1.0,
        gap_token: str = "▓",
    ):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.indel_score = indel_score
        self.gap_token = gap_token
        super().__init__()

    def align(self, gold: list[str], pred: list[str]) -> tuple[list[str], list[str]]:
        return needleman_wunsch(
            gold,
            pred,
            match_score=self.match_score,
            mismatch_score=self.mismatch_score,
            indel_score=self.indel_score,
            gap=self.gap_token,
        )
