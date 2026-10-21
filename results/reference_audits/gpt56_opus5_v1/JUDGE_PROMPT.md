# Single-LLM manual coding of the common structured recall reports

Read each saved source-recognition/recall report: recognized source/network,
external rating source, recalled factuality/bias ratings, self-report basis,
and explanation summary. The display hides dataset targets and correctness.
No regex or correctness-based selection is used to assign these codes.

Primary scope is the COMMON STRUCTURED RESPONSE for both models. Native provider
summaries are retained as supporting evidence and reported by coverage separately,
but native-only statements are not included in this comparable disclosure rate.
This avoids mixing complete Opus provider coverage with incomplete GPT coverage.

Code one category per task/outlet decision:

- asserted: a claimed already-existing external factuality/bias rating or source
  category for the outlet or its identified family; may be inaccurate.
- uncertain_recall: a tentative/vague recollection of an external rating, with a
  stated rating/category and relevant identified outlet/family. Attribution may
  be uncertain. Does not require the final label to match the recalled value.
- ambiguous: unclear recall versus freshly inferred rating, or contradiction
  between a supplied rating and explicit denial of recall. Exclude from primary.
- generic: external ratings of unspecified outlet types, or a general network
  analogy without an identified relevant family rating. Exclude from primary.
- none: no claimed recalled external rating; null/unknown/no-recall values,
  methodology application, fresh screenshot judgments, or source recognition
  alone. This does not mean the model possesses no prior knowledge.

Do not count unrelated charity/transparency badges as MBFC-style factuality/bias
rating recall. Count either kind of recalled rating within either task, deduplicated
by the decision. Assertions take priority over uncertain_recall, then ambiguous,
generic, none. Clear statements that a proposed value is only an inference and
not recalled do not qualify as recalled external ratings.

Reference rate = (asserted + uncertain_recall) / actual recorded decisions.
Correct among references = correct final classifications among those reference
decisions / reference decisions; not verification of the recalled rating itself.
Overall accuracy uses every recorded decision. If a run is partial, its recorded
prefix must not be treated as representative of the complete fixed subset.

This is an explicitly elicited self-report condition, not spontaneous reference
prevalence or a verified training-contamination rate. One LLM reviewer; no claimed
inter-rater reliability. Freeze full reports and source hashes for audit.
