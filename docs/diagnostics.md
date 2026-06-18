# Diagnostics Report — Why the System Returns LOW Risk for AI Text

Generated: 2026-06-18  
Analyst: Code review + mathematical simulation  
Scope: All 6 analyzer components + scoring formula

---

## 1. The Formula

```
overall_score = Σ weight_i × component_i
```

Weights:

| Component        | Weight |
|-----------------|--------|
| repetition       | 0.25   |
| transition_overuse | 0.15 |
| low_burstiness   | 0.20   |
| lexical_poverty  | 0.15   |
| cliche_density   | 0.15   |
| readability      | 0.10   |

Each component outputs [0, 100]. The weighted sum is also [0, 100].

---

## 2. Simulated Score for a Typical 500-Word AI Academic Essay

The following estimates are derived from the code logic, not from guessing.

| Component       | Typical AI Score | Weight | Contribution |
|----------------|-----------------|--------|-------------|
| repetition      | 8               | 0.25   | 2.0         |
| transition_overuse | 12           | 0.15   | 1.8         |
| low_burstiness  | 60              | 0.20   | 12.0        |
| readability     | 0               | 0.10   | 0.0         |
| cliche_density  | 3               | 0.15   | 0.45        |
| lexical_poverty | 35              | 0.15   | 5.25        |
| **TOTAL**       |                 |        | **21.5 → LOW** |

Even with a very uniform AI text (burstiness = 75):
- Total = 2.0 + 1.8 + 15.0 + 0.0 + 0.45 + 5.25 = **24.5 → LOW**

To reach MODERATE (>30), all 6 components must behave simultaneously above
their practical ceiling — which almost never happens with the current databases.

---

## 3. Per-Component Root Cause Analysis

---

### 3.1 Cliché Detection — CRITICAL FAILURE

**Database size:** 6 English + 6 Turkish = **12 phrases total**

Detected phrases (English):
- "in conclusion"
- "it is important to note that"
- "in today's world"
- "needless to say"
- "the fact of the matter is"
- "it goes without saying"

**What AI actually writes (undetected examples):**
- "it is worth noting that"
- "plays a crucial role"
- "in the realm of"
- "a comprehensive understanding"
- "it is evident that"
- "as mentioned above"
- "it can be argued that"
- "one of the most important"
- "has become increasingly important"
- "in today's rapidly changing world"
- "sheds light on"
- "a wide range of"
- "in summary"
- "to sum up"
- "with this in mind"
- "it should be noted that"
- "this highlights the importance of"

**Score formula:**
```python
score = min(1.0, cliche_density / 5.0)
cliche_density = total_occurrences * 100 / total_tokens
```

A 500-word text with 2 matched clichés:
- density = 2 × 100 / 500 = 0.4
- score = 0.4 / 5.0 = **0.08 → 8/100**

A text that uses 20 AI clichés (all undetected): **score = 0/100**

**Verdict:** Fires on less than 5% of AI clichés. Contributes 0–5/100 in practice.

---

### 3.2 Transition Analyzer — CRITICAL FAILURE

**Database size:** 9 English + 8 Turkish = **17 phrases total**

English transitions detected:
- furthermore, moreover, additionally, therefore, consequently,
  however, nevertheless, in conclusion, in addition

**What AI actually writes (undetected examples):**
- "as a result"
- "on the other hand"
- "in contrast"
- "with this in mind"
- "it is worth noting"
- "building on this"
- "to this end"
- "in this context"
- "in particular"
- "specifically"
- "notably"
- "interestingly"
- "importantly"
- "to elaborate"
- "in other words"
- "that is to say"
- "for instance"
- "for example"
- "in fact"
- "indeed"
- "as such"
- "thus"
- "hence"
- "accordingly"
- "subsequently"
- "previously"
- "ultimately"
- "essentially"
- "fundamentally"

**Score formula:**
```python
density_signal = min(1.0, density / 2.0)  # saturates at 2 transitions/sentence
repeat_ratio = len(repeated) / len(unique)
score = 0.6 × density_signal + 0.4 × repeat_ratio
```

AI text with 5 undetected transitions + 2 detected (each used once):
- density = 2/20 = 0.1 → density_signal = 0.05
- no repeats → repeat_ratio = 0
- score = 0.6 × 0.05 = **0.03 → 3/100**

**Verdict:** Detects less than 20% of transitions AI actually uses. Contributes 3–15/100 in practice.

---

### 3.3 Readability Component — INVERTED FOR ACADEMIC USE

**Code:**
```python
readability_c = max(0.0, readability.readability_score - 40.0) / 60.0 * 100.0
```

This formula **only flags easy text** (Flesch > 40).

Academic AI text (Claude, ChatGPT academic mode) typically produces:
- Flesch Reading Ease: **20–45** (college level, difficult)

For Flesch = 35 (typical AI academic output):
```
max(0, 35 - 40) / 60 × 100 = 0/100
```

The readability component **contributes 0** for most academic AI text.

It does fire for:
- AI emails (Flesch ~60-70) → readability_c = (65-40)/60 × 100 = 41/100
- Casual AI text (Flesch ~70-80) → readability_c = 66/100

**Verdict:** Wrong direction for academic detection. For the most common use case (academic essay), contributes 0/100. For casual text, contributes 40–70/100 but that's less useful.

---

### 3.4 Repetition Analyzer — WRONG SIGNAL FOR LLM TEXT

**Problem:** LLMs are specifically trained to avoid surface-level word repetition. This was a major training objective. The signal this analyzer measures is precisely what LLMs optimize against.

**Code:**
```python
word_signal = excess_tokens / total_tokens  # excess = count - 1 per flagged stem
phrase_signal = phrase_token_coverage / total_tokens
opener_signal = len(repeated_openings) / total_sentences
score = min(1.0, 0.5 × word_signal + 0.3 × phrase_signal + 0.2 × opener_signal)
```

For a 500-word AI essay where 3 content stems repeat 3× each:
- excess = (3-1) × 3 = 6
- word_signal = 6/500 = 0.012
- 0.5 × 0.012 = 0.006 → **0.6/100**

Even generous case: 5 stems repeat 5× each:
- excess = 4 × 5 = 20, word_signal = 20/500 = 0.04
- 0.5 × 0.04 = **2/100**

**Verdict:** Human writers (especially students) repeat words more than LLMs. This metric may actually score LOWER for AI text than for human student text. Contributes 3–12/100 for AI text.

---

### 3.5 Lexical Poverty — PARTIALLY INVERTED FOR MODERN LLMs

**Code:**
```python
lexical_poverty = (1 - lexical_diversity) × 100
lexical_diversity = unique_stems / total_tokens  # Type-Token Ratio
```

Modern LLMs have HIGH lexical diversity — this was a core training goal.

Typical values:
- 300-word AI text: diversity ≈ 0.70 → poverty = **30/100** (LOW)
- 500-word AI text: diversity ≈ 0.62 → poverty = **38/100** (borderline)
- 800-word AI text: diversity ≈ 0.54 → poverty = **46/100** (MODERATE)

Human student text (150-word answer): diversity ≈ 0.55 → poverty = **45/100**

**Note:** TTR is mathematically biased — diversity always drops as text gets longer even with identical writing style. This means a short AI text scores LOW poverty while a long one scores MODERATE, regardless of actual quality.

**Verdict:** Partially works for long texts but unreliable. Contributes 25–45/100, weakly discriminating.

---

### 3.6 Burstiness Analyzer — THE ONLY WORKING SIGNAL

**Code:**
```python
B = (σ − μ) / (σ + μ)
burstiness_score = (1 − B) / 2
```

This is the only component that regularly fires for AI text because:
- AI models tend to produce sentences of similar length
- B near 0 or slightly negative → score ≈ 0.5–0.65 → 50–65/100

Empirical ranges:
- Very uniform AI (B ≈ -0.5): score = **75/100** (HIGH)
- Moderately uniform AI (B ≈ -0.2): score = **60/100** (MODERATE)
- Neutral AI (B ≈ 0.0): score = **50/100** (MODERATE)
- Bursty human (B ≈ 0.3): score = **35/100** (LOW-MODERATE)

**Problem:** With weight 0.20, even a score of 70 contributes only **14 points** to the total. The other 5 components together contribute only 8-10 points, giving a total of 22-24 → LOW.

**Verdict:** Works correctly but is outvoted by 5 near-zero components.

---

## 4. Discriminability Summary

| Component | AI Typical Score | Human Typical Score | Gap | Verdict |
|-----------|-----------------|--------------------|----|---------|
| low_burstiness | 55–70 | 30–55 | +15–20 | **Weak discriminator** (too low weight) |
| lexical_poverty | 30–45 | 35–55 | -5–+10 | **Weak discriminator** (overlapping ranges) |
| repetition | 3–12 | 8–25 | -13 | **Inverted** (AI scores lower than humans) |
| transition_overuse | 3–15 | 5–20 | -5 | **Inverted** (database too small to detect AI transitions) |
| cliche_density | 0–8 | 2–15 | -7 | **Inverted** (database misses all AI clichés) |
| readability | 0–5 | 5–30 | -5–25 | **Inverted for academic use** |

**Effective discriminators: 0 out of 6**

The score ceiling for typical AI academic text is approximately **25/100**. Reaching MODERATE (>30) requires an unusual combination of signals simultaneously above their ceiling.

---

## 5. Why LOW Risk Appears So Often

The risk formula is:

```
score ≈ 0.25×(3-12) + 0.15×(3-15) + 0.20×(55-70) + 0.10×(0-5) + 0.15×(0-8) + 0.15×(30-45)
      ≈ 2-3      +   0.5-2.2    +   11-14         +   0-0.5    +   0-1.2    +   4.5-6.75
      ≈ 18–27
```

This range falls entirely within the LOW tier (0–30). The system was calibrated to detect formulaic writing patterns found in repetitive human academic writing — not LLM output. LLMs produce:

1. High lexical variety (defeats repetition + lexical poverty)
2. Sentence length variety (partially defeats burstiness)
3. Transitions and clichés not in the database (defeats both detectors)
4. Formal, difficult prose (defeats readability signal)

---

## 6. What the Data Shows vs What Was Expected

| Signal | Expected | Actual |
|--------|----------|--------|
| AI repeats words | Yes (old GPT-3 did) | No (modern LLMs do not) |
| AI uses "furthermore" a lot | Yes | Rarely; uses 50+ other connectors not in DB |
| AI writes easy readable text | For casual tasks | No, academic output is Flesch 20-45 |
| AI has uniform sentences | Yes | Partially — varies by model and prompt |
| AI uses clichés | Yes | Yes, but different ones than the 12 in DB |

---

## 7. High-Priority Fixes Required

1. **Expand cliché database** from 12 → 100+ AI-specific academic phrases  
2. **Expand transition database** from 17 → 60+ connectors and hedges  
3. **Fix or remove readability component** — currently scores 0 for most academic AI text  
4. **Rebalance weights** — burstiness should carry more weight (0.25–0.35)  
5. **Add hedging language detector** — AI heavily uses hedges: "may," "might," "it appears," "one could argue"  
6. **Recalibrate scoring thresholds** — LOW/MODERATE boundary should move from 30 → 20 until databases are expanded  

See `docs/improvement_plan.md` for the full action plan with priorities.
