"""Abstract base class contract for all text analyzers."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.models.analysis import AnalysisContext

T = TypeVar("T")


class BaseAnalyzer(ABC, Generic[T]):
    """Contract every analyzer in the pipeline must satisfy.

    Analyzers are pure functions over AnalysisContext: they receive a context,
    perform their analysis, and return a typed result. They must not mutate
    the context, perform I/O, or depend on mutable external state.

    Type Parameters:
        T: The result type produced by this analyzer.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier for this analyzer, e.g. 'word_stats'."""

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> T:
        """Run analysis on the context and return a typed result.

        Args:
            context: Immutable pipeline context produced by TokenizerService.

        Returns:
            Analysis result of type T.
        """
