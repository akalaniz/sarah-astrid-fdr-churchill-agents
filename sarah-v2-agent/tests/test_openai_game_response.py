from types import SimpleNamespace
import unittest

from app.core.openai_chat import SarahOpenAIClient


class _Responses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text='{"movement":"RIGHT","target":null,"fire_rounds":0}')


class GameResponseTests(unittest.TestCase):
    def test_reasoning_effort_is_scoped_to_requested_response(self) -> None:
        responses = _Responses()
        client = SarahOpenAIClient.__new__(SarahOpenAIClient)
        client._client = SimpleNamespace(responses=responses)
        answer = client.create_response(
            "gpt-5.6",
            [{"role": "user", "content": "game snapshot"}],
            reasoning_effort="none",
            max_output_tokens=80,
        )
        self.assertIn('"movement":"RIGHT"', answer)
        self.assertEqual(responses.request["reasoning"], {"effort": "none"})
        self.assertEqual(responses.request["max_output_tokens"], 80)

    def test_ordinary_response_does_not_add_reasoning_override(self) -> None:
        responses = _Responses()
        client = SarahOpenAIClient.__new__(SarahOpenAIClient)
        client._client = SimpleNamespace(responses=responses)
        client.create_response("gpt-5.6", [{"role": "user", "content": "ordinary chat"}])
        self.assertNotIn("reasoning", responses.request)
        self.assertNotIn("max_output_tokens", responses.request)


if __name__ == "__main__":
    unittest.main()
