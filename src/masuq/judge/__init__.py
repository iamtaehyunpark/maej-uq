"""Judge harness: prefix-conditional P(True) scoring with KV prefix sharing."""

from .backends import (
    HFGenerator,
    HFPrefixScorer,
    MockGenerator,
    MockPrefixScorer,
    OpenAIGenerator,
    PrefixScorer,
    ScoreTrace,
    TextGenerator,
    VerbalizedPrefixScorer,
)
from .evidence import Evidence, build_evidence, incremental_segments, turn_blocks
from .harness import (
    StepScore,
    TrajectoryScores,
    cost_summary,
    group_by_record,
    judge_corpus,
    judge_record,
    load_scores,
)
from .surrogate import MockSurrogateLM, SurrogateLM, SurrogateScore, score_corpus

__all__ = [
    "Evidence",
    "HFGenerator",
    "HFPrefixScorer",
    "MockGenerator",
    "MockPrefixScorer",
    "MockSurrogateLM",
    "OpenAIGenerator",
    "PrefixScorer",
    "ScoreTrace",
    "StepScore",
    "SurrogateLM",
    "SurrogateScore",
    "TextGenerator",
    "TrajectoryScores",
    "VerbalizedPrefixScorer",
    "build_evidence",
    "cost_summary",
    "group_by_record",
    "incremental_segments",
    "judge_corpus",
    "judge_record",
    "load_scores",
    "score_corpus",
    "turn_blocks",
]
