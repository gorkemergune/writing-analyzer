import dataclasses

import pytest

from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.services.tokenizer import (
    TokenizerService,
    _normalize_whitespace,
    _split_sentences_regex,
    _tokenize_en_nltk,
    _tokenize_en_regex,
    _tokenize_tr,
)

# ---------------------------------------------------------------------------
# Module-scoped fixture — zeyrek MorphAnalyzer loads once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerService:
    """Shared TokenizerService instance; avoids re-loading zeyrek per test."""
    return TokenizerService()


def _make_stub(*, nltk_ready: bool = False, porter=None, zeyrek=None) -> TokenizerService:
    """Bypass __init__ to build a TokenizerService with controlled internal state."""
    svc: TokenizerService = TokenizerService.__new__(TokenizerService)
    svc._nltk_ready = nltk_ready  # type: ignore[attr-defined]
    svc._porter = porter  # type: ignore[attr-defined]
    svc._zeyrek = zeyrek  # type: ignore[attr-defined]
    return svc


# ---------------------------------------------------------------------------
# Helper texts
# ---------------------------------------------------------------------------

_EN_TEXT = (
    "The impact of digital technology on modern education has been profound. "
    "Students today have access to a vast array of online resources. "
    "Educators must strike a careful balance between tradition and innovation."
)

_TR_TEXT = (
    "Dijital teknolojinin modern eğitim üzerindeki etkisi büyüktür. "
    "Öğrenciler çevrimiçi kaynaklara erişim imkânına sahiptir. "
    "Eğitimciler geleneksel ve yeni yaklaşımlar arasında denge kurmalıdır."
)


# ---------------------------------------------------------------------------
# _normalize_whitespace (pure function — no fixture needed)
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_strips_leading_whitespace(self):
        assert _normalize_whitespace("  hello") == "hello"

    def test_strips_trailing_whitespace(self):
        assert _normalize_whitespace("hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert _normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert _normalize_whitespace("hello\t\nworld") == "hello world"

    def test_empty_string_unchanged(self):
        assert _normalize_whitespace("") == ""

    def test_only_whitespace_becomes_empty(self):
        assert _normalize_whitespace("   \t  ") == ""

    def test_already_clean_text_unchanged(self):
        assert _normalize_whitespace("clean text here") == "clean text here"


# ---------------------------------------------------------------------------
# _split_sentences_regex (pure function — language-agnostic fallback)
# ---------------------------------------------------------------------------


class TestSplitSentencesRegex:
    def test_single_sentence_returns_one_element(self):
        result = _split_sentences_regex("Bu bir cümledir.")
        assert len(result) == 1
        assert result[0] == "Bu bir cümledir."

    def test_two_sentences_split_correctly(self):
        result = _split_sentences_regex("Birinci cümle. İkinci cümle.")
        assert len(result) == 2

    def test_exclamation_splits(self):
        result = _split_sentences_regex("Ne kadar güzel! Gerçekten harika.")
        assert len(result) == 2

    def test_question_splits(self):
        result = _split_sentences_regex("Bu ne anlama gelir? Açıklamak gerekir.")
        assert len(result) == 2

    def test_no_trailing_empty_strings(self):
        assert all(s for s in _split_sentences_regex("Cümle bir. Cümle iki."))

    def test_three_sentences(self):
        result = _split_sentences_regex("Birinci. İkinci. Üçüncü.")
        assert len(result) == 3

    def test_english_sentences_split(self):
        result = _split_sentences_regex("First sentence. Second sentence. Third.")
        assert len(result) == 3

    def test_no_boundary_returns_single_element(self):
        result = _split_sentences_regex("No punctuation here at all")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _tokenize_en_regex (pure function — NLTK-absent fallback)
# ---------------------------------------------------------------------------


class TestTokenizeEnRegex:
    def test_basic_tokenization(self):
        tokens = _tokenize_en_regex("The cat sat.")
        assert "the" in tokens
        assert "cat" in tokens
        assert "sat" in tokens

    def test_tokens_are_lowercase(self):
        tokens = _tokenize_en_regex("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_excluded(self):
        tokens = _tokenize_en_regex("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens

    def test_numbers_excluded(self):
        tokens = _tokenize_en_regex("There are 5 items.")
        assert "5" not in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize_en_regex("") == []

    def test_only_punctuation_returns_empty(self):
        assert _tokenize_en_regex("!!! ... ???") == []

    def test_returns_list(self):
        assert isinstance(_tokenize_en_regex("hello"), list)


# ---------------------------------------------------------------------------
# _tokenize_en_nltk (pure function — NLTK preferred path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not TokenizerService._prepare_nltk(),
    reason="NLTK punkt data not available",
)
class TestTokenizeEnNltk:
    def test_basic_tokenization(self):
        tokens = _tokenize_en_nltk("The cat sat.")
        assert "the" in tokens
        assert "cat" in tokens
        assert "sat" in tokens

    def test_tokens_are_lowercase(self):
        tokens = _tokenize_en_nltk("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_excluded(self):
        tokens = _tokenize_en_nltk("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_numbers_excluded(self):
        tokens = _tokenize_en_nltk("There are 5 items.")
        assert "5" not in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize_en_nltk("") == []

    def test_returns_list(self):
        assert isinstance(_tokenize_en_nltk("hello"), list)


# ---------------------------------------------------------------------------
# _tokenize_tr (pure function)
# ---------------------------------------------------------------------------


class TestTokenizeTurkish:
    def test_basic_tokenization(self):
        tokens = _tokenize_tr("Merhaba dünya.")
        assert "merhaba" in tokens
        assert "dünya" in tokens

    def test_tokens_are_lowercase(self):
        tokens = _tokenize_tr("Büyük Küçük")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_excluded(self):
        tokens = _tokenize_tr("Merhaba, dünya!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_turkish_diacritics_preserved(self):
        tokens = _tokenize_tr("çalışmak öğrenmek")
        assert "çalışmak" in tokens
        assert "öğrenmek" in tokens

    def test_dotless_i_preserved(self):
        tokens = _tokenize_tr("ışık")
        assert "ışık" in tokens

    def test_empty_string_returns_empty(self):
        assert _tokenize_tr("") == []

    def test_returns_list(self):
        assert isinstance(_tokenize_tr("merhaba"), list)


# ---------------------------------------------------------------------------
# TokenizerService properties
# ---------------------------------------------------------------------------


class TestServiceProperties:
    def test_nltk_available_returns_bool(self, tokenizer: TokenizerService):
        assert isinstance(tokenizer.nltk_available, bool)

    def test_zeyrek_available_returns_bool(self, tokenizer: TokenizerService):
        assert isinstance(tokenizer.zeyrek_available, bool)

    def test_stub_nltk_unavailable(self):
        svc = _make_stub(nltk_ready=False)
        assert svc.nltk_available is False

    def test_stub_zeyrek_unavailable(self):
        svc = _make_stub(zeyrek=None)
        assert svc.zeyrek_available is False

    def test_stub_zeyrek_available_when_set(self):
        svc = _make_stub(zeyrek=object())
        assert svc.zeyrek_available is True


# ---------------------------------------------------------------------------
# Fallback behaviour — NLTK absent
# ---------------------------------------------------------------------------


class TestNltkFallback:
    """Verify graceful degradation when NLTK is unavailable."""

    def test_build_context_succeeds_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx, AnalysisContext)

    def test_tokens_non_empty_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.tokens) > 0

    def test_sentences_non_empty_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.sentences) > 0

    def test_stems_equal_tokens_when_porter_absent(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert list(ctx.stems) == list(ctx.tokens)

    def test_stems_length_matches_tokens_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.stems) == len(ctx.tokens)

    def test_tokens_lowercase_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert all(t == t.lower() for t in ctx.tokens)

    def test_no_punctuation_tokens_without_nltk(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert "." not in ctx.tokens and "," not in ctx.tokens


# ---------------------------------------------------------------------------
# Fallback behaviour — zeyrek absent
# ---------------------------------------------------------------------------


class TestZeyrekFallback:
    """Verify graceful degradation when zeyrek is unavailable."""

    def test_build_context_succeeds_without_zeyrek(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert isinstance(ctx, AnalysisContext)

    def test_tokens_non_empty_without_zeyrek(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.tokens) > 0

    def test_stems_equal_tokens_when_zeyrek_absent(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert list(ctx.stems) == list(ctx.tokens)

    def test_stems_length_matches_tokens_without_zeyrek(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.stems) == len(ctx.tokens)

    def test_turkish_diacritics_preserved_without_zeyrek(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert any(any(c in t for c in "çğışöü") for t in ctx.tokens)

    def test_sentence_token_counts_consistent_without_zeyrek(self):
        svc = _make_stub(nltk_ready=False, porter=None, zeyrek=None)
        ctx = svc.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.sentence_token_counts) == len(ctx.sentences)


# ---------------------------------------------------------------------------
# TokenizerService.build_context — structure
# ---------------------------------------------------------------------------


class TestBuildContextStructure:
    def test_returns_analysis_context(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx, AnalysisContext)

    def test_raw_text_preserved(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert ctx.raw_text == _EN_TEXT

    def test_language_stored(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert ctx.language is Language.ENGLISH

    def test_document_type_stored(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ACADEMIC)
        assert ctx.document_type is DocumentType.ACADEMIC

    def test_cleaned_text_is_normalized(self, tokenizer: TokenizerService):
        messy = "  The  impact  of  technology. "
        ctx = tokenizer.build_context(messy, Language.ENGLISH, DocumentType.ESSAY)
        assert "  " not in ctx.cleaned_text
        assert not ctx.cleaned_text.startswith(" ")
        assert not ctx.cleaned_text.endswith(" ")

    def test_returned_context_is_frozen(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.raw_text = "modified"  # type: ignore[misc]

    def test_tokens_is_tuple(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx.tokens, tuple)

    def test_sentences_is_tuple(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx.sentences, tuple)

    def test_stems_is_tuple(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx.stems, tuple)

    def test_sentence_token_counts_is_tuple(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert isinstance(ctx.sentence_token_counts, tuple)


# ---------------------------------------------------------------------------
# TokenizerService.build_context — English content
# ---------------------------------------------------------------------------


class TestBuildContextEnglish:
    def test_tokens_non_empty(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.tokens) > 0

    def test_sentences_non_empty(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.sentences) > 0

    def test_stems_length_equals_tokens_length(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.stems) == len(ctx.tokens)

    def test_sentence_counts_length_equals_sentences_length(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.sentence_token_counts) == len(ctx.sentences)

    def test_tokens_are_lowercase(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert all(t == t.lower() for t in ctx.tokens)

    def test_tokens_contain_no_pure_punctuation(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert "." not in ctx.tokens
        assert "," not in ctx.tokens

    def test_three_sentences_detected(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.sentences) == 3

    def test_each_sentence_count_is_positive(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert all(c > 0 for c in ctx.sentence_token_counts)

    def test_known_word_appears_in_tokens(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert "impact" in ctx.tokens

    def test_stems_differ_from_tokens_for_inflected_words(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        assert ctx.stems != ctx.tokens or len(ctx.tokens) == 0


# ---------------------------------------------------------------------------
# TokenizerService.build_context — Turkish content
# ---------------------------------------------------------------------------


class TestBuildContextTurkish:
    def test_tokens_non_empty(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.tokens) > 0

    def test_sentences_non_empty(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.sentences) > 0

    def test_stems_length_equals_tokens_length(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.stems) == len(ctx.tokens)

    def test_sentence_counts_length_equals_sentences_length(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.sentence_token_counts) == len(ctx.sentences)

    def test_tokens_are_lowercase(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert all(t == t.lower() for t in ctx.tokens)

    def test_tokens_contain_no_punctuation(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert "." not in ctx.tokens
        assert "," not in ctx.tokens

    def test_turkish_diacritic_tokens_present(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert any(any(c in t for c in "çğışöüçğışöü") for t in ctx.tokens)

    def test_three_sentences_detected(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert len(ctx.sentences) == 3

    def test_known_word_appears_in_tokens(self, tokenizer: TokenizerService):
        ctx = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert "dijital" in ctx.tokens


# ---------------------------------------------------------------------------
# TokenizerService — edge cases
# ---------------------------------------------------------------------------


class TestBuildContextEdgeCases:
    def test_single_sentence_english(self, tokenizer: TokenizerService):
        text = "This is a single English sentence with several words in it."
        ctx = tokenizer.build_context(text, Language.ENGLISH, DocumentType.ESSAY)
        assert len(ctx.sentences) == 1
        assert len(ctx.tokens) > 0

    def test_leading_trailing_whitespace_cleaned(self, tokenizer: TokenizerService):
        text = "   " + _EN_TEXT + "   "
        ctx = tokenizer.build_context(text, Language.ENGLISH, DocumentType.ESSAY)
        assert not ctx.cleaned_text.startswith(" ")
        assert not ctx.cleaned_text.endswith(" ")
        assert ctx.raw_text == text

    def test_all_document_types_produce_valid_context(self, tokenizer: TokenizerService):
        for doc_type in DocumentType:
            ctx = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, doc_type)
            assert ctx.document_type is doc_type
            assert len(ctx.tokens) > 0

    def test_context_language_is_what_was_passed(self, tokenizer: TokenizerService):
        ctx_en = tokenizer.build_context(_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY)
        ctx_tr = tokenizer.build_context(_TR_TEXT, Language.TURKISH, DocumentType.ESSAY)
        assert ctx_en.language is Language.ENGLISH
        assert ctx_tr.language is Language.TURKISH
