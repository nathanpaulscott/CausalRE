# Event/State Span Annotation Guidelines

## 1. Flat Schema (No Hierarchies)
- Each span represents a **single coherent event or state**.
- No nested structures or trigger/argument decomposition.
- Annotate the **best contiguous span** that expresses the event/state.

---

## 2. Span Scope: Include Relevant Arguments
- Spans should include:
  - **Event trigger**
  - **Relevant arguments** (agent, theme, time, cause, etc.)
  - Minor interleaved noise is acceptable if it improves coherence.
- ✅ Example:
  > `"A few months after the hotel’s bombing the Government of Pakistan had reconstructed it"`

---

## 3. Span Boundary Preference
- Prefer the **widest natural span** that fully describes the event/state.
- Avoid fragmenting tightly integrated event components (e.g., time or agent).

---

## 4. When to Annotate Embedded Events Separately
Annotate an embedded span as a **separate event/state** if:
- It is **independently eventive**.
- It could be the **head of a causal relation**.
- It is **referable** elsewhere in the text.
- It conveys **meaning beyond being a modifier** of another event.

---

## 5. Overlapping Spans Are Allowed
- Overlapping spans are permitted where two events **share text** but represent **distinct semantic units**.
- Flat = no hierarchy, not disjointness.

---

## 6. What Not to Annotate
Do **not** annotate:
- Attributional or discourse-level elements (e.g., _"according to Xinhua"_).
- Standalone time references unless eventive (e.g.:
  - ✅ `"the explosion on 5 May"`
  - ❌ `"in 2008"`)

---

## 7. Causal Relations
- Annotate a causal link between spans **only when**:
  - One event **logically or directly** causes the other.
- Avoid:
  - Speculative, indirect, or metaphorical causality.
  - Weak implications without clear textual support.

---

## Heuristic for Ambiguous Cases
> _“Would this text span make sense as a row in a knowledge graph?”_  
✅ If yes → annotate.
