from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "ui" / "static"


class SarahInboxUiLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self.css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
        self.js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    def test_sarah_inbox_panel_renders_header_section(self) -> None:
        self.assertIn('id="agentInboxSection"', self.html)
        self.assertIn("<h2>Inter-agent inbox</h2>", self.html)
        self.assertIn('id="refreshAgentInboxButton"', self.html)
        self.assertIn('id="showAllAgentInboxButton"', self.html)
        self.assertIn('id="archiveAgentInboxButton"', self.html)
        self.assertIn('id="clearAgentInboxButton"', self.html)

    def test_sarah_inbox_panel_renders_send_note_section(self) -> None:
        self.assertIn('id="agentSendSection"', self.html)
        self.assertIn('id="agentMessageInput"', self.html)
        self.assertIn('id="sendAgentMessageButton"', self.html)
        self.assertIn("Send a note to Astrid...", self.html)
        self.assertIn(">Send to Astrid</button>", self.html)

    def test_sarah_inbox_panel_renders_message_list_section(self) -> None:
        self.assertIn('id="agentInboxPanel"', self.html)
        self.assertIn("agent-inbox-list", self.html)

    def test_send_note_section_appears_before_message_list_in_dom_order(self) -> None:
        self.assertLess(self.html.index('id="agentSendSection"'), self.html.index('id="agentInboxPanel"'))

    def test_send_textarea_is_not_nested_inside_message_list(self) -> None:
        inbox_div = _extract_div(self.html, "agentInboxPanel")
        self.assertNotIn("agentMessageInput", inbox_div)

    def test_send_button_is_not_nested_inside_message_list(self) -> None:
        inbox_div = _extract_div(self.html, "agentInboxPanel")
        self.assertNotIn("sendAgentMessageButton", inbox_div)

    def test_placeholder_and_button_target_astrid(self) -> None:
        self.assertIn('placeholder="Send a note to Astrid..."', self.html)
        self.assertIn("Send to Astrid", self.html)
        self.assertIn('to_agent: "Astrid"', self.js)

    def test_inbox_panel_uses_vertical_flex_layout(self) -> None:
        self.assertRegex(self.css, r"\.agent-panel\s*\{[^}]*display:\s*flex;[^}]*flex-direction:\s*column;")
        self.assertRegex(self.css, r"\.agent-tools\s*\{[^}]*flex:\s*0 0 auto;")
        self.assertRegex(self.css, r"\.agent-inbox-list\s*\{[^}]*flex:\s*1 1 auto;[^}]*overflow:\s*auto;")

    def test_sarah_still_uses_shared_agent_bus_for_inbox(self) -> None:
        agent_bus = (PROJECT_ROOT / "app" / "core" / "agent_bus.py").read_text(encoding="utf-8")
        self.assertIn('"shared_agent_bus"', agent_bus)
        self.assertNotIn('"shared_fdr_churchill_bus"', agent_bus)

    def test_no_fdr_or_churchill_paths_are_introduced(self) -> None:
        combined = "\n".join(
            [
                self.html,
                self.css,
                self.js,
                (PROJECT_ROOT / "app" / "ui" / "web_app.py").read_text(encoding="utf-8"),
            ]
        ).lower()
        self.assertNotIn("shared_fdr_churchill_bus", combined)
        self.assertNotIn("fdr_memory.jsonl", combined)
        self.assertNotIn("churchill_memory.jsonl", combined)
        self.assertNotIn("fdr-v1-agent", combined)
        self.assertNotIn("churchill-v1-agent", combined)


def _extract_div(html: str, element_id: str) -> str:
    pattern = re.compile(rf"<div\b[^>]*id=\"{re.escape(element_id)}\"[^>]*>.*?</div>", re.DOTALL)
    match = pattern.search(html)
    if not match:
        raise AssertionError(f"Could not find div #{element_id}")
    return match.group(0)


if __name__ == "__main__":
    unittest.main()
