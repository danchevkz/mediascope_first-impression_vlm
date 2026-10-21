"""Tests for provider-agnostic saved reasoning extraction."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_saved_run import extract_reasoning, reasoning_tokens


class SavedRunAnalysisTests(unittest.TestCase):
    def test_extracts_openai_summary(self):
        response = {"output": [{"type": "reasoning", "summary": [{"text": "visible evidence"}]}]}
        self.assertEqual(extract_reasoning(response), ("visible evidence", "openai_reasoning_summary", 1))

    def test_extracts_anthropic_thinking(self):
        response = {"content": [{"type": "thinking", "thinking": "publisher recall"},
                                {"type": "text", "text": "HIGH"}]}
        self.assertEqual(extract_reasoning(response), ("publisher recall", "anthropic_thinking", 1))

    def test_extracts_openrouter_reasoning(self):
        response = {"choices": [{"message": {"reasoning": "headline analysis"}}]}
        self.assertEqual(extract_reasoning(response), ("headline analysis", "openrouter_reasoning", 1))

    def test_reasoning_token_shapes(self):
        openai = {"usage": {"output_tokens_details": {"reasoning_tokens": 7}}}
        anthropic = {"usage": {"output_tokens_details": {"thinking_tokens": 9}}}
        self.assertEqual(reasoning_tokens(openai), 7)
        self.assertEqual(reasoning_tokens(anthropic), 9)


if __name__ == "__main__":
    unittest.main()
