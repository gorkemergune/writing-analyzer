"""Integration tests for AnalysisService — full pipeline with real dependencies."""

import pytest

from src.models.enums import DocumentType, Language, RiskLevel
from src.models.request import AnalysisRequest
from src.models.response import AnalysisReport
from src.services.analysis import AnalysisService
from src.services.language_detector import LanguageDetector

# ---------------------------------------------------------------------------
# Realistic text fixtures
# ---------------------------------------------------------------------------

# Well-written English academic text: varied vocabulary, varied sentence lengths,
# no clichés, no formulaic transitions. Expect LOW risk.
_GOOD_EN_TEXT = (
    "The relationship between working memory capacity and reading comprehension "
    "has been investigated extensively over the past three decades. "
    "Baddeley's multi-component model provides a useful framework: the phonological "
    "loop, visuospatial sketchpad, episodic buffer, and central executive interact "
    "when readers construct a mental model of a text. "
    "Crucially, individual differences in central-executive efficiency — not just "
    "storage capacity — predict comprehension outcomes across age groups. "
    "Short. "
    "Longitudinal studies suggest that early vocabulary breadth mediates this "
    "relationship, though the directionality remains contested in the literature."
)

# Well-written Turkish academic text: varied sentence lengths, no clichés,
# academic register. Expect LOW to MODERATE risk.
_GOOD_TR_TEXT = (
    "Çalışma belleği kapasitesi ile okuma anlayışı arasındaki ilişki, son otuz "
    "yılda kapsamlı biçimde araştırılmıştır. "
    "Baddeley'nin çok bileşenli modeli yararlı bir çerçeve sunar: sessel döngü, "
    "uzamsal-görsel çizim tahtası, bölümlü tampon ve merkezi yürütücü, okuyucular "
    "zihinsel bir metin modeli oluştururken etkileşime girer. "
    "Kısa. "
    "Merkezi yürütücü verimliliğindeki bireysel farklılıkların — salt depolama "
    "kapasitesinin değil — yaş gruplarında anlayış sonuçlarını yordadığı "
    "belirlenmiştir. "
    "Boylamsal çalışmalar, erken sözcük dağarcığı genişliğinin bu ilişkiye aracılık "
    "ettiğini ileri sürmektedir."
)

# Formulaic English text: clichés, repeated transitions, uniform sentences,
# repeated vocabulary. Expect HIGH or VERY_HIGH risk.
_FORMULAIC_EN_TEXT = (
    "In today's world, technology plays a crucial role in education. "
    "It is important to note that digital tools have transformed learning. "
    "Furthermore, technology has improved academic outcomes significantly. "
    "Furthermore, students benefit from technology every single day. "
    "Technology is important for students. Technology is very important indeed. "
    "In conclusion, technology is important for academic education today. "
    "Needless to say, technology will continue to be important for education. "
    "In conclusion, it goes without saying that technology matters greatly."
)

# Formulaic Turkish text: multiple clichés, repeated transitions, uniform
# sentence rhythm, limited vocabulary. Expect HIGH or VERY_HIGH risk.
_FORMULAIC_TR_TEXT = (
    "Günümüzde teknoloji büyük önem taşımaktadır. "
    "Bilindiği üzere dijital araçlar eğitimi dönüştürmüştür. "
    "Sonuç olarak teknoloji büyük önem taşımaktadır. "
    "Bunun altını çizmek gerekir teknoloji öğrencilere fayda sağlamaktadır. "
    "Teknoloji öğrenciler için büyük önem taşımaktadır gerçekten. "
    "Sonuç olarak teknoloji eğitim için büyük önem taşımaktadır kesinlikle. "
    "Günümüzde teknoloji teknoloji teknoloji eğitimde kullanılmaktadır. "
    "Yadsınamaz bir gerçektir ki teknoloji giderek büyük önem taşımaktadır."
)


def _make_service() -> AnalysisService:
    """Construct a default AnalysisService for integration tests."""
    return AnalysisService()


def _make_en_request(text: str, doc_type: DocumentType = DocumentType.ESSAY) -> AnalysisRequest:
    return AnalysisRequest(text=text, document_type=doc_type, language=Language.ENGLISH)


def _make_tr_request(text: str, doc_type: DocumentType = DocumentType.ACADEMIC) -> AnalysisRequest:
    return AnalysisRequest(text=text, document_type=doc_type, language=Language.TURKISH)


# ---------------------------------------------------------------------------
# TestEnglishPipeline
# ---------------------------------------------------------------------------


class TestEnglishPipeline:
    @pytest.fixture(scope="class")
    def report(self) -> AnalysisReport:
        return _make_service().analyze(_make_en_request(_GOOD_EN_TEXT))

    def test_completes_without_error(self, report: AnalysisReport):
        assert report is not None

    def test_language_is_english(self, report: AnalysisReport):
        assert report.language == Language.ENGLISH

    def test_document_type_preserved(self, report: AnalysisReport):
        assert report.document_type == DocumentType.ESSAY

    def test_word_stats_total_words_positive(self, report: AnalysisReport):
        assert report.word_stats.total_words > 0

    def test_word_stats_lexical_diversity_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.word_stats.lexical_diversity <= 1.0

    def test_sentence_stats_total_sentences_positive(self, report: AnalysisReport):
        assert report.sentence_stats.total_sentences > 0

    def test_sentence_stats_avg_length_positive(self, report: AnalysisReport):
        assert report.sentence_stats.avg_sentence_length > 0.0

    def test_readability_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.readability.readability_score <= 100.0

    def test_readability_has_grade_level(self, report: AnalysisReport):
        assert len(report.readability.grade_level) > 0

    def test_overall_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.overall_score <= 100.0

    def test_confidence_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.confidence <= 1.0

    def test_risk_level_valid(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in list(RiskLevel)

    def test_good_text_scores_low_to_moderate(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)

    def test_processing_time_positive(self, report: AnalysisReport):
        assert report.processing_time_ms > 0.0

    def test_highlights_is_list(self, report: AnalysisReport):
        assert isinstance(report.highlights, list)

    def test_suggestions_is_list(self, report: AnalysisReport):
        assert isinstance(report.suggestions, list)


# ---------------------------------------------------------------------------
# TestTurkishPipeline
# ---------------------------------------------------------------------------


class TestTurkishPipeline:
    @pytest.fixture(scope="class")
    def report(self) -> AnalysisReport:
        return _make_service().analyze(_make_tr_request(_GOOD_TR_TEXT))

    def test_completes_without_error(self, report: AnalysisReport):
        assert report is not None

    def test_language_is_turkish(self, report: AnalysisReport):
        assert report.language == Language.TURKISH

    def test_word_stats_total_words_positive(self, report: AnalysisReport):
        assert report.word_stats.total_words > 0

    def test_readability_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.readability.readability_score <= 100.0

    def test_readability_grade_contains_turkish_label(self, report: AnalysisReport):
        known_labels = (
            "İlkokul", "Ortaokul", "Lise", "Üniversite", "İleri Akademik",
        )
        assert any(label in report.readability.grade_level for label in known_labels)

    def test_overall_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.overall_score <= 100.0

    def test_risk_level_valid(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in list(RiskLevel)

    def test_good_text_scores_low_to_moderate(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)

    def test_processing_time_positive(self, report: AnalysisReport):
        assert report.processing_time_ms > 0.0


# ---------------------------------------------------------------------------
# TestFormulaicEnglish
# ---------------------------------------------------------------------------


class TestFormulaicEnglish:
    @pytest.fixture(scope="class")
    def report(self) -> AnalysisReport:
        return _make_service().analyze(_make_en_request(_FORMULAIC_EN_TEXT))

    def test_completes_without_error(self, report: AnalysisReport):
        assert report is not None

    def test_risk_level_is_high_or_very_high(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_cliches_detected(self, report: AnalysisReport):
        assert report.cliches.cliche_count > 0

    def test_in_conclusion_detected(self, report: AnalysisReport):
        assert "in conclusion" in report.cliches.detected_cliches

    def test_transitions_detected(self, report: AnalysisReport):
        assert report.transitions.transition_count > 0

    def test_furthermore_repeated(self, report: AnalysisReport):
        assert "furthermore" in report.transitions.repeated_transitions

    def test_explanations_generated(self, report: AnalysisReport):
        assert len(report.academic_risk.explanations) > 0

    def test_suggestions_generated(self, report: AnalysisReport):
        assert len(report.suggestions) > 0

    def test_highlights_generated(self, report: AnalysisReport):
        assert len(report.highlights) > 0

    def test_cliche_highlight_label_present(self, report: AnalysisReport):
        labels = {h.label for h in report.highlights}
        assert "cliche" in labels

    def test_cliche_component_score_elevated(self, report: AnalysisReport):
        assert report.academic_risk.component_scores.cliche_density > 40.0

    def test_transition_component_score_elevated(self, report: AnalysisReport):
        assert report.academic_risk.component_scores.transition_overuse > 40.0

    def test_explanations_use_probabilistic_language(self, report: AnalysisReport):
        for exp in report.academic_risk.explanations:
            assert "ai-generated" not in exp.lower()
            assert "definitely" not in exp.lower()


# ---------------------------------------------------------------------------
# TestFormulaicTurkish
# ---------------------------------------------------------------------------


class TestFormulaicTurkish:
    @pytest.fixture(scope="class")
    def report(self) -> AnalysisReport:
        return _make_service().analyze(_make_tr_request(_FORMULAIC_TR_TEXT))

    def test_completes_without_error(self, report: AnalysisReport):
        assert report is not None

    def test_risk_level_is_high_or_very_high(self, report: AnalysisReport):
        assert report.academic_risk.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_turkish_cliches_detected(self, report: AnalysisReport):
        assert report.cliches.cliche_count > 0

    def test_sonuc_olarak_detected(self, report: AnalysisReport):
        assert "sonuç olarak" in report.cliches.detected_cliches

    def test_gunumuzde_detected(self, report: AnalysisReport):
        assert "günümüzde" in report.cliches.detected_cliches

    def test_cliche_component_elevated(self, report: AnalysisReport):
        assert report.academic_risk.component_scores.cliche_density > 40.0

    def test_highlights_contain_turkish_cliche(self, report: AnalysisReport):
        assert len(report.highlights) > 0

    def test_explanations_generated(self, report: AnalysisReport):
        assert len(report.academic_risk.explanations) > 0

    def test_suggestions_generated(self, report: AnalysisReport):
        assert len(report.suggestions) > 0

    def test_overall_score_in_range(self, report: AnalysisReport):
        assert 0.0 <= report.academic_risk.overall_score <= 100.0


# ---------------------------------------------------------------------------
# TestLanguageAutoDetect
# ---------------------------------------------------------------------------


class TestLanguageAutoDetect:
    def test_explicit_english_bypasses_detector(self):
        called: list[bool] = []

        def _should_not_be_called(text: str) -> str:
            called.append(True)
            return "en"

        svc = AnalysisService(language_detector=LanguageDetector(_detect_fn=_should_not_be_called))
        svc.analyze(
            AnalysisRequest(text=_GOOD_EN_TEXT, language=Language.ENGLISH)
        )
        assert called == []

    def test_mock_detector_english_produces_en_report(self):
        svc = AnalysisService(
            language_detector=LanguageDetector(_detect_fn=lambda _: "en")
        )
        report = svc.analyze(AnalysisRequest(text=_GOOD_EN_TEXT))
        assert report.language == Language.ENGLISH

    def test_mock_detector_turkish_produces_tr_report(self):
        svc = AnalysisService(
            language_detector=LanguageDetector(_detect_fn=lambda _: "tr")
        )
        report = svc.analyze(AnalysisRequest(text=_GOOD_TR_TEXT))
        assert report.language == Language.TURKISH

    def test_unsupported_language_falls_back_to_english(self):
        svc = AnalysisService(
            language_detector=LanguageDetector(_detect_fn=lambda _: "fr")
        )
        report = svc.analyze(AnalysisRequest(text=_GOOD_EN_TEXT))
        # LanguageDetector falls back to Language.ENGLISH for unsupported codes
        assert report.language == Language.ENGLISH


# ---------------------------------------------------------------------------
# TestScoreInvariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,language,doc_type",
    [
        (_GOOD_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY),
        (_GOOD_TR_TEXT, Language.TURKISH, DocumentType.ACADEMIC),
        (_FORMULAIC_EN_TEXT, Language.ENGLISH, DocumentType.ESSAY),
        (_FORMULAIC_TR_TEXT, Language.TURKISH, DocumentType.ACADEMIC),
        (_GOOD_EN_TEXT, Language.ENGLISH, DocumentType.REPORT),
        (_GOOD_TR_TEXT, Language.TURKISH, DocumentType.ASSIGNMENT),
    ],
)
class TestScoreInvariants:
    def test_overall_score_in_range(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        assert 0.0 <= report.academic_risk.overall_score <= 100.0

    def test_confidence_in_range(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        assert 0.0 <= report.academic_risk.confidence <= 1.0

    def test_component_scores_in_range(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        cs = report.academic_risk.component_scores
        for val in (
            cs.repetition,
            cs.transition_overuse,
            cs.low_burstiness,
            cs.lexical_poverty,
            cs.cliche_density,
            cs.readability,
        ):
            assert 0.0 <= val <= 100.0

    def test_processing_time_is_nonnegative(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        assert report.processing_time_ms >= 0.0

    def test_highlights_do_not_overlap(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        for i in range(len(report.highlights) - 1):
            assert report.highlights[i].end <= report.highlights[i + 1].start

    def test_explanations_and_suggestions_aligned(self, text, language, doc_type):
        svc = _make_service()
        report = svc.analyze(AnalysisRequest(text=text, language=language, document_type=doc_type))
        assert len(report.academic_risk.explanations) == len(report.suggestions)


# ---------------------------------------------------------------------------
# TestDocumentTypes
# ---------------------------------------------------------------------------


class TestDocumentTypes:
    @pytest.mark.parametrize("doc_type", list(DocumentType))
    def test_each_document_type_accepted(self, doc_type: DocumentType):
        svc = _make_service()
        report = svc.analyze(
            AnalysisRequest(
                text=_GOOD_EN_TEXT,
                language=Language.ENGLISH,
                document_type=doc_type,
            )
        )
        assert report.document_type == doc_type
        assert 0.0 <= report.academic_risk.overall_score <= 100.0


# ---------------------------------------------------------------------------
# TestHighlightGeneration
# ---------------------------------------------------------------------------


class TestHighlightGeneration:
    @pytest.fixture(scope="class")
    def report(self) -> AnalysisReport:
        return _make_service().analyze(_make_en_request(_FORMULAIC_EN_TEXT))

    def test_highlight_start_lt_end(self, report: AnalysisReport):
        for h in report.highlights:
            assert h.start < h.end

    def test_highlight_offsets_reference_cleaned_text(self, report: AnalysisReport):
        for h in report.highlights:
            assert h.start >= 0
            assert h.end <= len(_FORMULAIC_EN_TEXT) + 50  # allow for whitespace normalization

    def test_cliche_label_in_highlights(self, report: AnalysisReport):
        labels = {h.label for h in report.highlights}
        assert "cliche" in labels

    def test_highlight_severity_is_valid_risk_level(self, report: AnalysisReport):
        for h in report.highlights:
            assert h.severity in list(RiskLevel)

    def test_suggestions_mention_cliche(self, report: AnalysisReport):
        combined = " ".join(report.suggestions).lower()
        assert "clich" in combined or "formulaic" in combined
