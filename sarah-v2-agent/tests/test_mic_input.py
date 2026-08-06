from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from app.core.config import load_settings
from app.ui.mic_input import TranscriptionError, transcribe_audio


class MicInputTests(unittest.TestCase):
    def test_transcription_errors_without_local_or_api_provider(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                audio_path = Path(tmp) / "sample.wav"
                audio_path.write_bytes(b"")
                settings = load_settings(env_file=Path(tmp) / ".env")

                with patch("app.ui.mic_input._transcribe_with_faster_whisper", return_value=None):
                    with patch("app.ui.mic_input._transcribe_with_whisper_cpp", return_value=None):
                        with self.assertRaises(TranscriptionError):
                            transcribe_audio(audio_path, settings)

    def test_transcription_uses_openai_fallback_when_local_is_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
                audio_path = Path(tmp) / "sample.wav"
                audio_path.write_bytes(b"")
                settings = load_settings(env_file=env_file)

                with patch("app.ui.mic_input._transcribe_with_faster_whisper", return_value=None):
                    with patch("app.ui.mic_input._transcribe_with_whisper_cpp", return_value=None):
                        with patch(
                            "app.ui.mic_input._transcribe_with_openai",
                            return_value="hello Sarah",
                        ) as openai_fallback:
                            transcript = transcribe_audio(audio_path, settings)

        self.assertEqual(transcript, "hello Sarah")
        openai_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
