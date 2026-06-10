import dataclasses

import pytest

from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language


def _make_context(**overrides: object) -> AnalysisContext:
    defaults: dict[str, object] = dict(
        raw_text="Some raw text for testing purposes here.",
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text="Some raw text for testing purposes here.",
        tokens=("some", "raw", "text", "for", "testing"),
        sentences=("Some raw text for testing purposes here.",),
        stems=("some", "raw", "text", "for", "test"),
        sentence_token_counts=(5,),
    )
    return AnalysisContext(**{**defaults, **overrides})


class TestAnalysisContextConstruction:
    def test_basic_construction(self):
        ctx = _make_context()
        assert ctx.raw_text == "Some raw text for testing purposes here."
        assert ctx.language is Language.ENGLISH
        assert ctx.document_type is DocumentType.ESSAY

    def test_cleaned_text_stored(self):
        ctx = _make_context(cleaned_text="clean version")
        assert ctx.cleaned_text == "clean version"

    def test_turkish_language_and_document_type(self):
        ctx = _make_context(language=Language.TURKISH, document_type=DocumentType.ACADEMIC)
        assert ctx.language is Language.TURKISH
        assert ctx.document_type is DocumentType.ACADEMIC

    def test_all_document_types_accepted(self):
        for doc_type in DocumentType:
            ctx = _make_context(document_type=doc_type)
            assert ctx.document_type is doc_type

    def test_multiple_sentences(self):
        sentences = ("First sentence.", "Second sentence.", "Third sentence.")
        ctx = _make_context(sentences=sentences, sentence_token_counts=(2, 2, 2))
        assert len(ctx.sentences) == 3
        assert ctx.sentence_token_counts == (2, 2, 2)

    def test_empty_tokens_valid(self):
        ctx = _make_context(tokens=(), stems=(), sentences=(), sentence_token_counts=())
        assert ctx.tokens == ()
        assert len(ctx.tokens) == 0

    def test_token_values_preserved(self):
        tokens = ("the", "quick", "brown", "fox")
        ctx = _make_context(tokens=tokens)
        assert ctx.tokens == tokens

    def test_stem_values_preserved(self):
        stems = ("the", "quick", "brown", "fox")
        ctx = _make_context(stems=stems)
        assert ctx.stems == stems

    def test_sentence_token_counts_preserved(self):
        counts = (5, 10, 3)
        sentences = ("s1 s2 s3 s4 s5.", "s " * 10, "s1 s2 s3.")
        ctx = _make_context(sentences=sentences, sentence_token_counts=counts)
        assert ctx.sentence_token_counts == counts

    def test_raw_text_differs_from_cleaned(self):
        ctx = _make_context(raw_text="  Raw  Text  ", cleaned_text="Raw Text")
        assert ctx.raw_text == "  Raw  Text  "
        assert ctx.cleaned_text == "Raw Text"


class TestAnalysisContextTypes:
    def test_tokens_type_is_tuple(self):
        ctx = _make_context()
        assert isinstance(ctx.tokens, tuple)

    def test_sentences_type_is_tuple(self):
        ctx = _make_context()
        assert isinstance(ctx.sentences, tuple)

    def test_stems_type_is_tuple(self):
        ctx = _make_context()
        assert isinstance(ctx.stems, tuple)

    def test_sentence_token_counts_type_is_tuple(self):
        ctx = _make_context()
        assert isinstance(ctx.sentence_token_counts, tuple)

    def test_raw_text_type_is_str(self):
        ctx = _make_context()
        assert isinstance(ctx.raw_text, str)

    def test_cleaned_text_type_is_str(self):
        ctx = _make_context()
        assert isinstance(ctx.cleaned_text, str)

    def test_language_type_is_language_enum(self):
        ctx = _make_context()
        assert isinstance(ctx.language, Language)

    def test_document_type_is_document_type_enum(self):
        ctx = _make_context()
        assert isinstance(ctx.document_type, DocumentType)


class TestAnalysisContextImmutability:
    def test_cannot_reassign_raw_text(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.raw_text = "modified"  # type: ignore[misc]

    def test_cannot_reassign_language(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.language = Language.TURKISH  # type: ignore[misc]

    def test_cannot_reassign_document_type(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.document_type = DocumentType.ACADEMIC  # type: ignore[misc]

    def test_cannot_reassign_tokens(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.tokens = ("new", "tokens")  # type: ignore[misc]

    def test_cannot_reassign_sentences(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.sentences = ("new sentence.",)  # type: ignore[misc]

    def test_cannot_reassign_stems(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.stems = ("new", "stems")  # type: ignore[misc]

    def test_cannot_reassign_sentence_token_counts(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.sentence_token_counts = (99,)  # type: ignore[misc]

    def test_cannot_reassign_cleaned_text(self):
        ctx = _make_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.cleaned_text = "other"  # type: ignore[misc]


class TestAnalysisContextDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(AnalysisContext)

    def test_is_frozen(self):
        assert AnalysisContext.__dataclass_params__.frozen is True

    def test_field_count(self):
        assert len(dataclasses.fields(AnalysisContext)) == 8

    def test_all_required_field_names(self):
        names = {f.name for f in dataclasses.fields(AnalysisContext)}
        assert names == {
            "raw_text",
            "language",
            "document_type",
            "cleaned_text",
            "tokens",
            "sentences",
            "stems",
            "sentence_token_counts",
        }

    def test_equality_same_values(self):
        ctx_a = _make_context()
        ctx_b = _make_context()
        assert ctx_a == ctx_b

    def test_inequality_different_language(self):
        ctx_a = _make_context(language=Language.ENGLISH)
        ctx_b = _make_context(language=Language.TURKISH)
        assert ctx_a != ctx_b
