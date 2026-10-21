from eval_finetuned_rationales import parse_label, rationale_prompt


def test_parse_label_accepts_rationale_final_line():
    assert parse_label("Visible sourcing is weak.\nLABEL: MIXED", ["LOW", "MIXED", "HIGH"]) == "MIXED"


def test_parse_label_accepts_direct_label_for_adapter_fallback():
    assert parse_label(" high ", ["LOW", "MIXED", "HIGH"]) == "HIGH"


def test_parse_label_accepts_markdown_bold_final_line():
    assert parse_label("Evidence.\n**LABEL: HIGH**", ["LOW", "MIXED", "HIGH"]) == "HIGH"


def test_parse_label_rejects_label_only_in_explanation():
    assert parse_label("I considered HIGH but chose something else.", ["LOW", "MIXED", "HIGH"]) is None


def test_prompt_requests_basis_without_prompting_rating_recall():
    prompt = rationale_prompt("factuality")
    assert "Briefly explain the basis" in prompt
    assert "remember" not in prompt.lower()
    assert "LABEL: <label>" in prompt
