"""Bilingual tokenizer service for the Academic Writing Auditor."""

import re
from typing import Any

from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language

# ---------------------------------------------------------------------------
# Module-level optional import — NLTK is desired but not required.
# The flag is checked before any nltk call so the app never raises
# ModuleNotFoundError if the package is absent.
# ---------------------------------------------------------------------------
try:
    import nltk
    import nltk.stem

    _NLTK_IMPORTABLE: bool = True
except ImportError:
    _NLTK_IMPORTABLE = False

# ---------------------------------------------------------------------------
# Compiled patterns used by the regex-based fallback pipelines.
# ---------------------------------------------------------------------------

# Turkish alphabetic characters including all diacritics: ç ğ ı ö ş ü + uppercase.
_WORD_TR = re.compile(r"[a-zA-ZçğışöüÇĞİÖŞÜ]+", re.UNICODE)

# Plain ASCII alphabetic fallback for English when NLTK is absent.
_WORD_EN = re.compile(r"[a-zA-Z]+")

# Sentence boundary: any .!? followed by one or more whitespace characters.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_WHITESPACE = re.compile(r"\s+")


class TokenizerService:
    """Builds an AnalysisContext from raw text using language-specific processing.

    Preferred pipelines (when NLP packages are present):
        English: NLTK sentence splitter → NLTK word tokenizer → Porter stemmer.
        Turkish: regex sentence splitter → Unicode word extractor →
            zeyrek morphological lemmatizer.

    Degraded pipelines (automatic fallback when packages are absent):
        English (no NLTK): regex sentence splitter → regex word extractor →
            no stemming (identity).
        Turkish (no zeyrek): regex sentence splitter → regex word extractor →
            no stemming (identity).

    The service never raises due to missing NLP packages. All optional
    dependencies are guarded and replaced with equivalent pure-Python logic.
    """

    def __init__(self) -> None:
        self._nltk_ready: bool = self._prepare_nltk()
        self._porter: Any = self._load_porter()
        self._zeyrek: Any = self._load_zeyrek()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_nltk() -> bool:
        """Ensure NLTK punkt data is present and return whether it is usable.

        Attempts to locate punkt and punkt_tab tokenizer data. If either is
        missing a download is attempted. Returns False when NLTK is not
        installed, when data cannot be located, or when the download itself
        fails (e.g. no network access). In all failure cases the service
        falls back to regex tokenization transparently.
        """
        if not _NLTK_IMPORTABLE:
            return False
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    nltk.data.find(f"tokenizers/{resource}")
                except Exception:
                    return False
        return True

    @staticmethod
    def _load_porter() -> Any:
        """Return a PorterStemmer if NLTK is available, else None."""
        if not _NLTK_IMPORTABLE:
            return None
        try:
            return nltk.stem.PorterStemmer()
        except Exception:
            return None

    @staticmethod
    def _load_zeyrek() -> Any:
        """Return a zeyrek MorphAnalyzer if the package is available, else None.

        Catches all exceptions — not only ImportError — because MorphAnalyzer()
        can raise OSError or RuntimeError when bundled model files are missing
        or corrupt. Any failure causes the service to continue without
        morphological stemming.
        """
        try:
            import zeyrek  # type: ignore[import-untyped]

            return zeyrek.MorphAnalyzer()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_context(
        self,
        raw_text: str,
        language: Language,
        document_type: DocumentType,
    ) -> AnalysisContext:
        """Build an immutable AnalysisContext from raw text.

        Args:
            raw_text: Unprocessed input text from the request.
            language: Resolved language that determines the tokenization strategy.
            document_type: Document genre carried unchanged into the context.

        Returns:
            Frozen AnalysisContext ready for the analyzer pipeline.
        """
        cleaned = _normalize_whitespace(raw_text)
        sentences = self._split_sentences(cleaned, language)
        tokens = self._tokenize(cleaned, language)
        stems = self._stem(tokens, language)
        sentence_token_counts = tuple(
            len(self._tokenize(s, language)) for s in sentences
        )
        return AnalysisContext(
            raw_text=raw_text,
            language=language,
            document_type=document_type,
            cleaned_text=cleaned,
            tokens=tuple(tokens),
            sentences=tuple(sentences),
            stems=tuple(stems),
            sentence_token_counts=sentence_token_counts,
        )

    @property
    def nltk_available(self) -> bool:
        """True when NLTK is installed and punkt data is loaded and ready."""
        return self._nltk_ready

    @property
    def zeyrek_available(self) -> bool:
        """True when zeyrek is installed and MorphAnalyzer initialised successfully."""
        return self._zeyrek is not None

    # ------------------------------------------------------------------
    # Internal pipeline — sentence splitting
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str, language: Language) -> list[str]:
        if language == Language.ENGLISH and self._nltk_ready:
            return nltk.sent_tokenize(text)
        return _split_sentences_regex(text)

    # ------------------------------------------------------------------
    # Internal pipeline — word tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str, language: Language) -> list[str]:
        if language == Language.TURKISH:
            return _tokenize_tr(text)
        if self._nltk_ready:
            return _tokenize_en_nltk(text)
        return _tokenize_en_regex(text)

    # ------------------------------------------------------------------
    # Internal pipeline — stemming / lemmatization
    # ------------------------------------------------------------------

    def _stem(self, tokens: list[str], language: Language) -> list[str]:
        if language == Language.ENGLISH:
            if self._porter is not None:
                return [self._porter.stem(t) for t in tokens]
            return list(tokens)
        return self._lemmatize_tr(tokens)

    def _lemmatize_tr(self, tokens: list[str]) -> list[str]:
        """Lemmatize Turkish tokens via zeyrek; return tokens unchanged on any failure."""
        if self._zeyrek is None:
            return list(tokens)
        results: list[str] = []
        for token in tokens:
            try:
                parses = self._zeyrek.lemmatize(token)
                if parses and parses[0][1]:
                    results.append(parses[0][1][0].lower())
                else:
                    results.append(token)
            except Exception:
                results.append(token)
        return results


# ---------------------------------------------------------------------------
# Module-level pure functions — independently testable, no class state.
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace sequences and strip leading/trailing space."""
    return _WHITESPACE.sub(" ", text).strip()


def _split_sentences_regex(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries (language-agnostic fallback)."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize_en_nltk(text: str) -> list[str]:
    """Tokenize English text with NLTK: lowercased, alphabetic tokens only."""
    return [t.lower() for t in nltk.word_tokenize(text) if t.isalpha()]


def _tokenize_en_regex(text: str) -> list[str]:
    """Tokenize English text with regex: lowercased ASCII alphabetic tokens.

    Used when NLTK is unavailable. Less accurate for contractions and
    hyphenated words but never raises.
    """
    return _WORD_EN.findall(text.lower())


def _tokenize_tr(text: str) -> list[str]:
    """Tokenize Turkish text: alphabetic tokens including all Turkish diacritics."""
    return _WORD_TR.findall(text.lower())
