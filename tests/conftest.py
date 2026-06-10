"""Shared test fixtures for the Academic Writing Auditor test suite."""

import pytest

from src.models.analysis import AnalysisContext
from src.models.enums import DocumentType, Language
from src.models.request import AnalysisRequest

# ---------------------------------------------------------------------------
# Raw text constants reused by both fixtures and request objects
# ---------------------------------------------------------------------------

EN_ESSAY_TEXT = (
    "The impact of digital technology on modern education has been profound and far-reaching. "
    "Educational institutions across the world are increasingly adopting digital tools to "
    "enhance the learning experience. Students today have access to a vast array of online "
    "resources, interactive platforms, and collaborative tools. The integration of technology "
    "in classrooms has transformed the way teachers deliver instruction and how students engage "
    "with academic content. Research suggests that well-implemented technology can improve "
    "student engagement and academic outcomes. However, the digital divide remains a significant "
    "challenge, as not all students have equal access to devices and reliable internet connections. "
    "Educators must therefore strike a careful balance between leveraging digital tools and "
    "maintaining traditional pedagogical approaches that foster analytical skills."
)

TR_ESSAY_TEXT = (
    "Dijital teknolojinin modern eğitim üzerindeki etkisi derin ve geniş kapsamlı olmuştur. "
    "Dünyanın dört bir yanındaki eğitim kurumları, öğrenme deneyimini geliştirmek için giderek "
    "daha fazla dijital araçlar benimsemektedir. Günümüz öğrencileri, çevrimiçi kaynaklara, "
    "etkileşimli platformlara ve işbirliği araçlarına erişim imkânına sahiptir. Sınıflarda "
    "teknolojinin entegrasyonu, öğretmenlerin ders verme biçimini ve öğrencilerin akademik "
    "içerikle nasıl etkileşime girdiğini dönüştürmüştür. Araştırmalar, iyi uygulanmış "
    "teknolojinin öğrenci katılımını ve akademik başarıyı artırabileceğini öne sürmektedir. "
    "Eğitimciler bu nedenle dijital araçlardan yararlanmak ile geleneksel pedagojik "
    "yaklaşımları sürdürmek arasında dikkatli bir denge kurmalıdır."
)

# ---------------------------------------------------------------------------
# Rich text constants for analyzer tests that need more token variety
# ---------------------------------------------------------------------------

EN_RICH_TEXT = (
    "Language shapes not only how we communicate but also how we perceive and categorize "
    "reality. Philosophers have long debated whether the structure of language constrains "
    "the thoughts its speakers can form. Some researchers maintain that thought is largely "
    "independent of language, while others argue that linguistic categories influence "
    "cognition in measurable ways. The relationship between language and thought remains "
    "one of the most contested questions in cognitive science."
)

TR_RICH_TEXT = (
    "Dil, yalnızca nasıl iletişim kurduğumuzu değil, gerçekliği nasıl algıladığımızı "
    "ve kategorize ettiğimizi de şekillendirir. Filozoflar, dilin yapısının konuşanların "
    "ifade edebileceği düşünceleri kısıtlayıp kısıtlamadığını uzun süredir tartışmaktadır. "
    "Bazı araştırmacılar düşüncenin büyük ölçüde dilden bağımsız olduğunu savunurken, "
    "diğerleri dilsel kategorilerin bilişi ölçülebilir biçimlerde etkilediğini öne sürmektedir. "
    "Dil ile düşünce arasındaki ilişki, bilişsel bilimin en tartışmalı sorularından biri olmaya "
    "devam etmektedir."
)


@pytest.fixture
def en_essay_text() -> str:
    """Return a sample English academic essay paragraph."""
    return EN_ESSAY_TEXT


@pytest.fixture
def tr_essay_text() -> str:
    """Return a sample Turkish academic essay paragraph."""
    return TR_ESSAY_TEXT


@pytest.fixture
def valid_en_request() -> AnalysisRequest:
    """Return a valid English analysis request with explicit language."""
    return AnalysisRequest(
        text=EN_ESSAY_TEXT,
        document_type=DocumentType.ESSAY,
        language=Language.ENGLISH,
    )


@pytest.fixture
def valid_tr_request() -> AnalysisRequest:
    """Return a valid Turkish analysis request with explicit language."""
    return AnalysisRequest(
        text=TR_ESSAY_TEXT,
        document_type=DocumentType.ACADEMIC,
        language=Language.TURKISH,
    )


@pytest.fixture
def en_analysis_context() -> AnalysisContext:
    """Return a pre-built English AnalysisContext for analyzer unit tests."""
    tokens = (
        "the", "impact", "of", "digital", "technology", "on", "modern",
        "education", "has", "been", "profound", "students", "today", "have",
        "access", "to", "a", "vast", "array", "of", "online", "resources",
    )
    sentences = (
        "The impact of digital technology on modern education has been profound.",
        "Students today have access to a vast array of online resources.",
        "The integration of technology in classrooms has transformed instruction.",
    )
    stems = (
        "the", "impact", "of", "digit", "technolog", "on", "modern",
        "educ", "has", "been", "profound", "student", "today", "have",
        "access", "to", "a", "vast", "arrai", "of", "onlin", "resourc",
    )
    return AnalysisContext(
        raw_text=EN_ESSAY_TEXT,
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text=EN_ESSAY_TEXT.strip(),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(10, 12, 9),
    )


@pytest.fixture
def tr_analysis_context() -> AnalysisContext:
    """Return a pre-built Turkish AnalysisContext for analyzer unit tests."""
    tokens = (
        "dijital", "teknolojinin", "modern", "eğitim", "üzerindeki",
        "etkisi", "derin", "ve", "geniş", "kapsamlı", "olmuştur",
        "eğitim", "kurumları", "dijital", "araçlar", "benimsemektedir",
    )
    sentences = (
        "Dijital teknolojinin modern eğitim üzerindeki etkisi derin ve geniş kapsamlı olmuştur.",
        "Eğitim kurumları dijital araçlar benimsemektedir.",
        "Araştırmalar teknolojinin başarıyı artırabileceğini öne sürmektedir.",
    )
    stems = (
        "dijital", "teknoloji", "modern", "eğitim", "üzerindeki",
        "etki", "derin", "ve", "geniş", "kapsamlı", "ol",
        "eğitim", "kurum", "dijital", "araç", "benimsemek",
    )
    return AnalysisContext(
        raw_text=TR_ESSAY_TEXT,
        language=Language.TURKISH,
        document_type=DocumentType.ACADEMIC,
        cleaned_text=TR_ESSAY_TEXT.strip(),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(11, 5, 8),
    )


@pytest.fixture
def en_rich_context() -> AnalysisContext:
    """Richer English AnalysisContext for analyzers that need more token variety.

    23 tokens, 3 repeated stems ("digit" x3, "educ" x2, "student" x2) → 19 unique stems.
    Unique stems: {digit, technolog, has, transform, modern, educ, tool, enhanc, the,
                   learn, experi, student, access, onlin, resourc, adopt, approach,
                   improv, outcom}
    lexical_diversity = 19/23
    avg_word_length = 168/23  (surface-form character sum = 168)
    """
    tokens = (
        "digital", "technology", "has", "transformed", "modern", "education",
        "digital", "tools", "enhance", "the", "learning", "experience",
        "students", "access", "online", "resources", "educators", "adopt",
        "digital", "approaches", "improve", "student", "outcomes",
    )
    stems = (
        "digit", "technolog", "has", "transform", "modern", "educ",
        "digit", "tool", "enhanc", "the", "learn", "experi",
        "student", "access", "onlin", "resourc", "educ", "adopt",
        "digit", "approach", "improv", "student", "outcom",
    )
    sentences = (
        "Digital technology has transformed modern education.",
        "Digital tools enhance the learning experience.",
        "Students access online resources and educators adopt digital approaches.",
        "Improve student outcomes.",
    )
    return AnalysisContext(
        raw_text=EN_RICH_TEXT,
        language=Language.ENGLISH,
        document_type=DocumentType.ACADEMIC,
        cleaned_text=EN_RICH_TEXT.strip(),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(6, 6, 9, 3),
    )


@pytest.fixture
def tr_rich_context() -> AnalysisContext:
    """Richer Turkish AnalysisContext reflecting agglutinative morphology.

    20 tokens, 3 repeated stems ("dijital" x2, "teknoloji" x2, "eğitim" x2) → 17 unique.
    Turkish surface forms are longer on average than English (~9.2 chars vs ~7.3).
    lexical_diversity = 17/20 = 0.85
    avg_word_length = 184/20 = 9.2  (surface-form character sum = 184)
    """
    tokens = (
        "dijital", "teknoloji", "eğitimi", "dönüştürmüştür", "öğrenciler",
        "çevrimiçi", "kaynaklara", "erişim", "dijital", "araçlar",
        "eğitimciler", "yeni", "yaklaşımlar", "benimsemektedir", "teknoloji",
        "başarıyı", "artırmaktadır", "eğitim", "kalitesi", "gelişmektedir",
    )
    stems = (
        "dijital", "teknoloji", "eğitim", "dönüştür", "öğrenci",
        "çevrimiçi", "kaynak", "erişim", "dijital", "araç",
        "eğitimci", "yeni", "yaklaşım", "benimse", "teknoloji",
        "başarı", "artır", "eğitim", "kalite", "geliş",
    )
    sentences = (
        "Dijital teknoloji eğitimi dönüştürmüştür.",
        "Öğrenciler çevrimiçi kaynaklara erişim sağlamaktadır.",
        "Dijital araçlar eğitimciler için yeni yaklaşımlar benimsemektedir.",
        "Teknoloji başarıyı artırmaktadır.",
        "Eğitim kalitesi gelişmektedir.",
    )
    return AnalysisContext(
        raw_text=TR_RICH_TEXT,
        language=Language.TURKISH,
        document_type=DocumentType.ACADEMIC,
        cleaned_text=TR_RICH_TEXT.strip(),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(4, 5, 6, 3, 3),
    )


@pytest.fixture
def uniform_sentence_context() -> AnalysisContext:
    """English context where every sentence has exactly 8 tokens.

    Designed to expose zero-variance behaviour in SentenceStatisticsAnalyzer
    and perfect-uniformity behaviour in BurstinessAnalyzer.

    sentence_token_counts = (8, 8, 8, 8, 8)
    total_sentences=5, avg=8.0, variance=0.0, min=8, max=8
    """
    tokens = (
        "the", "cat", "sat", "quietly", "on", "the", "wooden", "fence",
        "she", "walked", "slowly", "down", "the", "long", "dark", "road",
        "he", "opened", "the", "heavy", "door", "with", "great", "care",
        "the", "old", "man", "read", "his", "favorite", "worn", "book",
        "a", "bright", "star", "shone", "above", "the", "quiet", "town",
    )
    stems = (
        "the", "cat", "sat", "quietli", "on", "the", "wooden", "fenc",
        "she", "walk", "slowli", "down", "the", "long", "dark", "road",
        "he", "open", "the", "heavi", "door", "with", "great", "care",
        "the", "old", "man", "read", "his", "favorit", "worn", "book",
        "a", "bright", "star", "shone", "abov", "the", "quiet", "town",
    )
    sentences = (
        "The cat sat quietly on the wooden fence.",
        "She walked slowly down the long dark road.",
        "He opened the heavy door with great care.",
        "The old man read his favorite worn book.",
        "A bright star shone above the quiet town.",
    )
    return AnalysisContext(
        raw_text=" ".join(sentences),
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text=" ".join(sentences),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(8, 8, 8, 8, 8),
    )


@pytest.fixture
def variable_sentence_context() -> AnalysisContext:
    """English context with highly variable sentence lengths.

    Demonstrates the contrast with uniform_sentence_context for both
    SentenceStatisticsAnalyzer and BurstinessAnalyzer.

    sentence_token_counts = (2, 18, 3, 20, 2)
    total_sentences=5, total_tokens=45
    mean = 45/5 = 9.0
    variance = (49+81+36+121+49)/5 = 336/5 = 67.2
    min=2, max=20
    """
    tokens = (
        # sentence 1 — 2 tokens
        "yes", "indeed",
        # sentence 2 — 18 tokens
        "the", "rapid", "advancement", "of", "contemporary", "digital",
        "technology", "in", "educational", "settings", "has", "fundamentally",
        "transformed", "how", "students", "learn", "and", "engage",
        # sentence 3 — 3 tokens
        "no", "one", "knows",
        # sentence 4 — 20 tokens
        "furthermore", "the", "integration", "of", "evidence", "based",
        "pedagogical", "strategies", "alongside", "collaborative", "digital",
        "platforms", "has", "been", "empirically", "demonstrated", "to",
        "improve", "student", "outcomes",
        # sentence 5 — 2 tokens
        "absolutely", "not",
    )
    stems = tokens  # identity: sentence stats do not use stems
    sentences = (
        "Yes indeed.",
        "The rapid advancement of contemporary digital technology in educational "
        "settings has fundamentally transformed how students learn and engage.",
        "No one knows.",
        "Furthermore the integration of evidence based pedagogical strategies "
        "alongside collaborative digital platforms has been empirically "
        "demonstrated to improve student outcomes.",
        "Absolutely not.",
    )
    return AnalysisContext(
        raw_text=" ".join(sentences),
        language=Language.ENGLISH,
        document_type=DocumentType.ESSAY,
        cleaned_text=" ".join(sentences),
        tokens=tokens,
        sentences=sentences,
        stems=stems,
        sentence_token_counts=(2, 18, 3, 20, 2),
    )
