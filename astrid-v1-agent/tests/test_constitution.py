import unittest
from unittest.mock import patch

from app.persona.constitution import (
    DEFAULT_CONSTITUTION_PATH,
    build_constitution_prompt_section,
    load_sarah_constitution,
)


class ConstitutionTests(unittest.TestCase):
    def test_adult_intimacy_mode_loads_from_yaml(self) -> None:
        constitution = load_sarah_constitution()
        mode = constitution.adult_intimacy_mode

        self.assertIn("Astrid Bach, not Sarah", constitution.identity)
        self.assertIn("Discovery Mars mission propulsion engineer", constitution.identity)
        self.assertIn("never speak as Sarah", constitution.never)
        self.assertIn("never inherit Sarah's command trauma center", constitution.never)
        self.assertIn("speak like Astrid Bach, a brilliant adult woman and propulsion engineer", constitution.always)
        self.assertEqual(mode["model_source"], "Astrid 2_0_a.docx")
        self.assertIn("warm adult flirtation", mode["allowed"])
        self.assertIn("multiple choice intimacy menus", mode["forbidden"])
        self.assertIn("Astrid speaks naturally in prose.", mode["style_rules"])

    def test_adult_intimacy_mode_is_in_prompt_section(self) -> None:
        prompt_section = build_constitution_prompt_section()

        self.assertIn("adult_intimacy_mode:", prompt_section)
        self.assertIn("model_source: Astrid 2_0_a.docx", prompt_section)
        self.assertIn("Astrid answers as Astrid.", prompt_section)

    def test_fallback_parser_handles_adult_intimacy_mode(self) -> None:
        with patch.dict("sys.modules", {"yaml": None}):
            constitution = load_sarah_constitution(DEFAULT_CONSTITUTION_PATH)

        self.assertEqual(constitution.adult_intimacy_mode["model_source"], "Astrid 2_0_a.docx")
        self.assertIn("humor and heat", constitution.adult_intimacy_mode["allowed"])


if __name__ == "__main__":
    unittest.main()
