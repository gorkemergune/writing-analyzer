# Academic Writing Auditor

A bilingual (English + Turkish) text analysis platform that evaluates writing quality and identifies patterns associated with formulaic or repetitive writing.

> The system does not claim AI authorship with certainty. It surfaces evidence-based writing quality signals.

## Features

- Word statistics — token count, lexical diversity, average word length
- Sentence statistics — length distribution, variance
- Repetition detection — repeated words, phrases, sentence openers
- Transition analysis, burstiness, readability, cliché detection *(in progress)*
- Academic risk score aggregating all signals *(in progress)*

## Stack

- Python 3.13 · FastAPI · Pydantic v2 · Streamlit (MVP)
- NLP: NLTK (English), Zeyrek (Turkish)
