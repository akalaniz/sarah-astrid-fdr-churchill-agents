import unittest
from unittest.mock import patch

from app.demo import DEMO_PROMPTS, _safe_print


class DemoTests(unittest.TestCase):
    def test_demo_prompts_are_the_required_three(self) -> None:
        self.assertEqual(
            DEMO_PROMPTS,
            [
                "Sarah, who are you?",
                "Darwin scares me.",
                "Iran tests a nuclear weapon. Give me COAs.",
            ],
        )

    def test_safe_print_accepts_unicode_text(self) -> None:
        with patch("builtins.print"):
            _safe_print("arrow -> unicode arrow \u2192")


if __name__ == "__main__":
    unittest.main()
