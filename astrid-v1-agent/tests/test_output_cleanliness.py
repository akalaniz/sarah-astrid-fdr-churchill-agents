import logging
import unittest

from app.core.logging import configure_logging
from app.core.text_output import normalize_sarah_output
from app.core.prompt_builder import build_safety_boundaries


class OutputCleanlinessTests(unittest.TestCase):
    def test_http_client_loggers_are_warning_or_higher(self) -> None:
        configure_logging("INFO")

        self.assertGreaterEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertGreaterEqual(logging.getLogger("httpcore").level, logging.WARNING)
        self.assertGreaterEqual(logging.getLogger("openai").level, logging.WARNING)

    def test_normalize_sarah_output_removes_markdown_bold_markers(self) -> None:
        self.assertEqual(
            normalize_sarah_output("Sarah: **important** and **dry**."),
            "Sarah: important and dry.",
        )

    def test_normalize_sarah_output_removes_markdown_heading_hashes(self) -> None:
        self.assertEqual(
            normalize_sarah_output("# Situation\n## COAs\nSarah keeps it plain."),
            "Situation\nCOAs\nSarah keeps it plain.",
        )

    def test_normalize_sarah_output_removes_inline_hashes_and_single_asterisks(self) -> None:
        self.assertEqual(
            normalize_sarah_output("Use tag #Mars and *plain* emphasis if Alex asks."),
            "Use tag Mars and plain emphasis if Alex asks.",
        )

    def test_prompt_tells_sarah_not_to_use_markdown_formatting(self) -> None:
        boundaries = build_safety_boundaries()

        self.assertIn("do not use hashtag/hash characters", boundaries)
        self.assertIn("do not use asterisk characters", boundaries)
        self.assertIn("do not use Markdown heading markers", boundaries)


if __name__ == "__main__":
    unittest.main()
