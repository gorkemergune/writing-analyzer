import pytest

from src.models.enums import DocumentType, Language, RiskLevel


class TestLanguage:
    def test_english_value(self):
        assert Language.ENGLISH == "en"

    def test_turkish_value(self):
        assert Language.TURKISH == "tr"

    def test_is_str_subclass(self):
        assert isinstance(Language.ENGLISH, str)
        assert isinstance(Language.TURKISH, str)

    def test_all_values(self):
        assert {lang.value for lang in Language} == {"en", "tr"}

    def test_from_string_english(self):
        assert Language("en") is Language.ENGLISH

    def test_from_string_turkish(self):
        assert Language("tr") is Language.TURKISH

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            Language("fr")

    def test_membership(self):
        assert Language.ENGLISH in Language
        assert Language.TURKISH in Language

    def test_count(self):
        assert len(Language) == 2

    def test_usable_as_dict_key(self):
        mapping = {Language.ENGLISH: "en", Language.TURKISH: "tr"}
        assert mapping[Language.ENGLISH] == "en"

    def test_string_equality(self):
        assert Language.ENGLISH == "en"
        assert Language.TURKISH == "tr"


class TestDocumentType:
    def test_essay_value(self):
        assert DocumentType.ESSAY == "essay"

    def test_academic_value(self):
        assert DocumentType.ACADEMIC == "academic"

    def test_email_value(self):
        assert DocumentType.EMAIL == "email"

    def test_report_value(self):
        assert DocumentType.REPORT == "report"

    def test_assignment_value(self):
        assert DocumentType.ASSIGNMENT == "assignment"

    def test_all_members_are_str(self):
        for doc_type in DocumentType:
            assert isinstance(doc_type, str)

    def test_all_values_present(self):
        assert {dt.value for dt in DocumentType} == {
            "essay",
            "academic",
            "email",
            "report",
            "assignment",
        }

    def test_from_string_essay(self):
        assert DocumentType("essay") is DocumentType.ESSAY

    def test_from_string_academic(self):
        assert DocumentType("academic") is DocumentType.ACADEMIC

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            DocumentType("thesis")

    def test_count(self):
        assert len(DocumentType) == 5

    def test_usable_as_dict_key(self):
        mapping = {dt: dt.value for dt in DocumentType}
        assert mapping[DocumentType.REPORT] == "report"


class TestRiskLevel:
    def test_low_value(self):
        assert RiskLevel.LOW == "low"

    def test_moderate_value(self):
        assert RiskLevel.MODERATE == "moderate"

    def test_high_value(self):
        assert RiskLevel.HIGH == "high"

    def test_very_high_value(self):
        assert RiskLevel.VERY_HIGH == "very_high"

    def test_all_members_are_str(self):
        for level in RiskLevel:
            assert isinstance(level, str)

    def test_all_values_present(self):
        assert {level.value for level in RiskLevel} == {
            "low",
            "moderate",
            "high",
            "very_high",
        }

    def test_from_string_low(self):
        assert RiskLevel("low") is RiskLevel.LOW

    def test_from_string_very_high(self):
        assert RiskLevel("very_high") is RiskLevel.VERY_HIGH

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            RiskLevel("critical")

    def test_count(self):
        assert len(RiskLevel) == 4

    def test_declaration_order(self):
        levels = list(RiskLevel)
        assert levels[0] is RiskLevel.LOW
        assert levels[-1] is RiskLevel.VERY_HIGH

    def test_usable_as_dict_key(self):
        thresholds = {RiskLevel.LOW: 30, RiskLevel.MODERATE: 55}
        assert thresholds[RiskLevel.LOW] == 30
