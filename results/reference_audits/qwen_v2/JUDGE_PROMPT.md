# Conservative manual rating-reference judge — version 2

Apply to the identical structured recognition reports from every model, including
GPT and Opus. This clarification addresses generic network analogies being counted
as uncertain recall. Preserve earlier annotations and results as version 1; do not
silently replace them or compare different coding versions as if equivalent.

## Review inputs and blinding

Read recognized outlet/network, external rating source, both recalled-rating
fields, self-report basis, and explanation. Hide dataset targets, final correctness,
and aggregate performance while coding. Do not use regex, non-null fields, final
label agreement, or the model's influence flag as the adjudicator.

Native provider reasoning is retained for supplementary analysis, with coverage
reported separately. It is outside the common structured-report comparison.
Refused/filtered/invalid/missing reports are unobserved, not negative references.

## Required annotation

For each valid task/outlet report, persist:

- `category`: asserted, uncertain_recall, ambiguous, generic, or none.
- `rating_scope`: outlet, named_family, generic_type, unclear, or none.
- `family_link`: identified_by_response, tentative_or_inferred, not_applicable.
- `evidence`: exact saved-text excerpt(s) supporting the decision.
- `note`: brief explanation of the attribution and uncertainty when relevant.

## Decision rules

1. Count a claimed existing external factuality/bias rating only if it is clearly
   attributed to the particular outlet or a **named relevant family**. Either
   kind of rating can qualify within either task, once per decision.
2. An unqualified claim is asserted; tentative, weak, vague, or qualified memory
   is uncertain_recall. A confidently remembered parent rating can be asserted
   while the outlet-to-parent link is tentative; retain those distinctions.
3. A recalled rating for a named parent may qualify despite an inferred family
   link, but publish tentative-family counts separately. Recognition of a design
   alone does not supply either a remembered rating or a verified affiliation.
4. Statements that "similar automated sites," "this type of local network," or
   "pink-slime sites" are generally rated right/mixed are generic unless the
   rating claim itself is anchored to the named family. Mentioning Metric Media
   or MBFC elsewhere does not turn a generic analogy into named-family recall.
5. Null rating fields do not force none: a clear, anchored recollection in the
   basis/explanation can qualify. Conversely, populated fields do not qualify
   if the supplied value is explicitly only a fresh inference or conflicts with
   a denial of recall; contradictory/unclear claims are ambiguous.
6. Source identity, ideology, ownership, conservative funding, familiar layout,
   application of MBFC methodology, and the final predicted label are not
   recalled external ratings by themselves.
7. Inaccurate or possibly hallucinated external-rating claims can qualify as
   disclosed claims; this judge does not independently validate ratings or memory.

## Report all distinctions

- Primary broad disclosure rate: asserted + uncertain_recall, with anchored
  outlet/named-family scopes only, divided by valid reviewed structured reports.
- Asserted-only sensitivity rate, using that same denominator.
- Separate outlet-specific, identified-family, and tentative-family counts.
- Generic/ambiguous cases excluded and enumerated, with evidence available.
- Correct final predictions among references; this is not recalled-rating truth.
- Overall classification accuracy on the fixed 212-outlet task population and
  valid-report/native-reasoning coverage. Provider refusals count as incorrect
  classifications but remain unobserved recall disclosures.

One LLM's repeated review is not independent inter-rater validation. These are
explicitly elicited disclosures, not verified training-contamination estimates.
