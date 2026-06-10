"""API request model for the Academic Writing Auditor."""

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import DocumentType, Language


class AnalysisRequest(BaseModel):
    """Incoming request for a text analysis operation.

    Attributes:
        text: The text to analyze. Minimum 50 characters.
        document_type: Genre of the document. Affects scoring weights.
        language: Language of the text. Auto-detected when None.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "The rapid advancement of artificial intelligence has transformed "
                    "modern industries. Organizations are increasingly adopting machine "
                    "learning algorithms to optimize their operations and decision-making "
                    "processes across various sectors of the economy."
                ),
                "document_type": "academic",
                "language": "en",
            }
        }
    )

    text: str = Field(
        ...,
        min_length=50,
        description="The text to analyze. Minimum 50 characters.",
    )
    document_type: DocumentType = Field(
        default=DocumentType.ESSAY,
        description="The genre of the document.",
    )
    language: Language | None = Field(
        default=None,
        description="Language of the text. Auto-detected when None.",
    )
