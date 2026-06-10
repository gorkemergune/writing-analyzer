"""Enumeration types for the Academic Writing Auditor domain."""

from enum import Enum


class Language(str, Enum):
    """Supported analysis languages."""

    ENGLISH = "en"
    TURKISH = "tr"


class DocumentType(str, Enum):
    """Supported input document types for context-aware scoring."""

    ESSAY = "essay"
    ACADEMIC = "academic"
    EMAIL = "email"
    REPORT = "report"
    ASSIGNMENT = "assignment"


class RiskLevel(str, Enum):
    """Academic risk classification tiers.

    Score ranges:
        LOW: 0–30
        MODERATE: 31–55
        HIGH: 56–75
        VERY_HIGH: 76–100
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
