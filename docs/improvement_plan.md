# Improvement Plan — Fixing the LOW Risk Problem

Generated: 2026-06-18  
Based on: docs/diagnostics.md

---

## Root Cause Summary

5 of 6 analysis components generate near-zero scores for typical AI academic text.
Only burstiness works, but with weight 0.20 it cannot carry the total above 30 (LOW).

Effective discriminators currently: **0 out of 6**

---

## Priority 1 — Keep and Fix (No New Architecture)

### 1.1 Expand Cliché Database (CRITICAL)
**Current:** 12 phrases  
**Target:** 120+ phrases  
**Estimated impact:** cliche_density score: 0–8 → 15–60 for AI text

Add to `_EN_CLICHES` in `src/analyzers/cliche.py`:

```
it is worth noting that
it should be noted that
it can be argued that
one of the most important
plays a crucial role
in the realm of
has become increasingly important
a wide range of
it is evident that
it is clear that
as mentioned above / as mentioned earlier
it is important to understand
with this in mind
building upon this
to this end
in this context
sheds light on
a comprehensive understanding
in today's rapidly changing world
it is undeniable that
there is no doubt that
it is widely accepted that
serves as a foundation
lays the groundwork
at the forefront of
it is imperative to
moving forward
going forward
last but not least
to put it simply
all things considered
on the whole
as a matter of fact
for all intents and purposes
at the end of the day
```

And Turkish:
```
bu bağlamda
önem kazanmaktadır
dikkate değer bir husus
göz ardı edilemez
ön plana çıkmaktadır
bu çerçevede
önem arz etmektedir
bu minvalde
değerlendirmek gerekmektedir
söz konusu
ele alındığında
irdelendiğinde
vurgulanmalıdır
belirtmek gerekir ki
dikkat çekmektedir
üzerinde durmak gerekmektedir
incelendiğinde
değerlendirildiğinde
farkında olmak gerekmektedir
tartışmasız bir gerçektir
```

### 1.2 Expand Transition Database (HIGH)
**Current:** 17 phrases  
**Target:** 70+ phrases  
**Estimated impact:** transition_overuse score: 3–15 → 20–55 for AI text

Add to `_EN_TRANSITIONS` in `src/analyzers/transition.py`:

```
as a result
on the other hand
in contrast
by contrast
in particular
specifically
notably
importantly
interestingly
in light of this
with this in mind
to this end
in this regard
with respect to
in terms of
it is worth noting
that said
to elaborate
in other words
to clarify
for instance
for example
in fact
indeed
as such
thus
hence
accordingly
subsequently
ultimately
essentially
fundamentally
to summarize
to conclude
in summary
overall
taken together
on balance
by extension
in addition to this
beyond this
building on this
above all
```

And Turkish:
```
nitekim
üstelik
öte yandan
diğer taraftan
bu nedenle
bu sebeple
buna ek olarak
bununla beraber
bu doğrultuda
özellikle
başta
öncelikle
son olarak
bu itibarla
bu açıdan
bu perspektiften
buna karşın
buna rağmen
oysa
halbuki
```

### 1.3 Fix Readability Component Direction (HIGH)
**Current behavior:** Flags EASY text (Flesch > 40). Returns 0 for difficult academic text.  
**Problem:** AI academic writing has Flesch 20–45 → scores 0/100.

**Option A — Remove** (simplest): Set weight to 0, redistribute to burstiness.  
**Option B — Invert**: Flag text that is TOO DIFFICULT (overly complex vocabulary pattern typical of AI formal writing).  
**Option C — Replace**: Measure vocabulary sophistication (avg syllables per word) instead.

Recommendation: **Option A** for now (remove), pending research on a better signal.

Change in `src/scoring/weights.py`:
```python
ScoringWeights(
    repetition=0.25,
    transition_overuse=0.20,  # +0.05
    low_burstiness=0.25,      # +0.05
    lexical_poverty=0.15,
    cliche_density=0.15,
    readability=0.00,         # removed
)
```

### 1.4 Rebalance Weights (MEDIUM)
After expanding databases, recalibrate weights:

```
burstiness:      0.25  (up from 0.20 — most reliable signal)
transition:      0.20  (up from 0.15 — now with 70+ phrases)
cliche_density:  0.20  (up from 0.15 — now with 120+ phrases)
lexical_poverty: 0.15  (keep)
repetition:      0.15  (down from 0.25 — inverted signal for LLM)
readability:     0.05  (down from 0.10 — signal is weak/noisy)
```

### 1.5 Lower Risk Thresholds Temporarily (QUICK FIX)
While databases are being expanded, lower the LOW/MODERATE boundary:

Current: LOW ≤ 30  
**Proposed: LOW ≤ 22** (based on simulated AI score of 18–27)

This is a calibration patch, not a real fix. Remove once component scores improve.

---

## Priority 2 — New Analyzers

### 2.1 Hedging Language Analyzer (HIGH IMPACT)

AI text uses hedging language at high density:
- "may," "might," "could," "it appears," "it seems," "one could argue"
- "it is possible that," "there is evidence to suggest," "arguably"

Human student text: hedges rarely, uses direct assertions.

Implementation:
- New file: `src/analyzers/hedging.py`
- Dataset: ~40 English + ~30 Turkish hedging phrases
- Score: hedges per 100 words, normalized to [0,1]
- Add to scorer with weight ~0.15

Estimated discrimination gap: human mean ~10, LLM mean ~35 → gap +25

### 2.2 Sentence Structure Uniformity (MEDIUM IMPACT)

AI tends to produce sentences with similar syntactic structure:
- Subject + verb + object → elaborate clause
- Consistent use of comma + connector pattern
- Predictable clause depth

Proxy metric (no full parser needed):
- Measure distribution of sentence-ending punctuation
- Measure ratio of compound sentences (contain ", and/but/which/that")
- Count sentences starting with the same POS pattern

### 2.3 Vocabulary Sophistication Flatness (MEDIUM IMPACT)

AI uses consistent register throughout. Humans mix formal and informal words.

Metric: standard deviation of average syllable count across sentence groups.
Low variance = uniform register = AI signal.

---

## Priority 3 — Structural Changes (Phase 2)

### 3.1 Paragraph-Level Analysis
AI structures essays in predictable patterns:
- Intro paragraph: defines topic + states thesis
- Body paragraphs: topic sentence + evidence + analysis + transition
- Conclusion: restates thesis + broader implications

Detecting this pattern would be a strong signal.

### 3.2 Discourse Coherence Score
AI essays are often over-coherent — every sentence connects to the previous one
with explicit discourse markers. Human writing has gaps, jumps, and implicit transitions.

### 3.3 N-gram Frequency Proxy
High-frequency n-grams from AI output logs could be used as a lookup table.
This requires a reference corpus but would catch model-specific phrases.

---

## Expected Performance After Priority 1 Fixes

| Component | Current AI Score | After Fix |
|-----------|-----------------|-----------|
| cliche_density | 0–8 | 15–60 |
| transition_overuse | 3–15 | 20–55 |
| low_burstiness | 55–70 | 55–70 (unchanged) |
| lexical_poverty | 25–45 | 25–45 (unchanged) |
| repetition | 3–12 | 3–12 (unchanged) |
| readability | 0–5 | 0 (removed) |

With new weights:
```
overall ≈ 0.15×8 + 0.20×35 + 0.25×62 + 0.15×35 + 0.20×35 + 0.05×0
         = 1.2 + 7.0 + 15.5 + 5.25 + 7.0 + 0
         = 35.95 → MODERATE
```

Target: AI academic text → MODERATE to HIGH (40–65)  
Target: Human student text → LOW to MODERATE (15–35)

---

## Implementation Order

1. Expand cliché database (1 hour)
2. Expand transition database (1 hour)
3. Rebalance weights / remove readability (30 minutes)
4. Run benchmark to verify improvement
5. Add hedging analyzer (2–3 hours)
6. Re-run benchmark and recalibrate thresholds
7. Write test cases for new databases

---

## What NOT to Change

- **Burstiness formula**: mathematically sound, already works
- **Lexical poverty (TTR)**: contributes some signal, worth keeping
- **Repetition formula**: the math is correct but LLMs simply don't repeat; keep at reduced weight
- **Clean Architecture**: no changes to service/analyzer structure needed
- **API contract**: `AnalysisReport` shape should not change for clients
