from __future__ import annotations

from dataclasses import dataclass
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from app.core.config import Settings


@dataclass(frozen=True)
class MicrophoneConfig:
    sample_rate: int = 16_000
    channels: int = 1
    frame_duration_seconds: float = 0.1
    silence_threshold: float = 0.012
    silence_seconds: float = 1.25
    initial_silence_timeout_seconds: float = 10.0
    max_record_seconds: float = 60.0


class MicrophoneInputError(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


def capture_and_transcribe(settings: Settings) -> str:
    audio_path = capture_microphone_audio()
    try:
        return transcribe_audio(audio_path, settings).strip()
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass


def capture_microphone_audio(config: MicrophoneConfig | None = None) -> Path:
    config = config or MicrophoneConfig()
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise MicrophoneInputError(
            "Microphone capture requires optional packages. Run: "
            'python -m pip install -e ".[mic]"'
        ) from exc

    frame_count = int(config.sample_rate * config.frame_duration_seconds)
    frames = []
    speech_started = False
    silent_for = 0.0
    recorded_for = 0.0

    print("Listening... speak now. Pause for a moment to stop; Ctrl+C also stops recording.")
    try:
        with sd.InputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype="float32",
        ) as stream:
            while recorded_for < config.max_record_seconds:
                data, _overflowed = stream.read(frame_count)
                frames.append(data.copy())
                recorded_for += config.frame_duration_seconds

                rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
                if rms >= config.silence_threshold:
                    speech_started = True
                    silent_for = 0.0
                    continue

                if speech_started:
                    silent_for += config.frame_duration_seconds
                    if silent_for >= config.silence_seconds:
                        break
                elif recorded_for >= config.initial_silence_timeout_seconds:
                    break
    except KeyboardInterrupt:
        print()

    if not frames:
        raise MicrophoneInputError("No microphone audio was captured.")

    audio = np.concatenate(frames, axis=0)
    if not speech_started and _rms(audio) < config.silence_threshold:
        raise MicrophoneInputError("No speech detected.")

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    return _write_wav(pcm16, config.sample_rate, config.channels)


def transcribe_audio(audio_path: Path, settings: Settings) -> str:
    local_text = _transcribe_with_faster_whisper(audio_path, settings)
    if local_text:
        return local_text

    whisper_cpp_text = _transcribe_with_whisper_cpp(audio_path, settings)
    if whisper_cpp_text:
        return whisper_cpp_text

    if not settings.openai_api_key:
        raise TranscriptionError(
            "No local Whisper transcriber was found and OPENAI_API_KEY is not set. "
            "Install faster-whisper, configure whisper.cpp, or set OPENAI_API_KEY for OpenAI transcription."
        )
    return _transcribe_with_openai(audio_path, settings)


def _transcribe_with_faster_whisper(audio_path: Path, settings: Settings) -> str | None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    model = WhisperModel(settings.whisper_model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(audio_path), vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def _transcribe_with_whisper_cpp(audio_path: Path, settings: Settings) -> str | None:
    exe = settings.whisper_cpp_exe or shutil.which("whisper-cli")
    if not exe:
        return None
    if not settings.whisper_cpp_model:
        raise TranscriptionError("WHISPER_CPP_MODEL must point to a whisper.cpp model file.")

    command = [
        exe,
        "-m",
        settings.whisper_cpp_model,
        "-f",
        str(audio_path),
        "-nt",
        "-otxt",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or "whisper.cpp transcription failed.")

    output_path = audio_path.with_suffix(audio_path.suffix + ".txt")
    if output_path.exists():
        text = output_path.read_text(encoding="utf-8", errors="ignore").strip()
        output_path.unlink(missing_ok=True)
        return text
    return result.stdout.strip()


def _transcribe_with_openai(audio_path: Path, settings: Settings) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TranscriptionError("OpenAI SDK is not installed. Run: python -m pip install -e .") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=settings.transcription_model,
            file=audio_file,
        )

    text = getattr(transcript, "text", None)
    if text:
        return text
    if isinstance(transcript, str):
        return transcript
    raise TranscriptionError("OpenAI transcription response did not include text.")


def _write_wav(audio, sample_rate: int, channels: int) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="sarah-mic-", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return path


def _rms(audio) -> float:
    if audio.size == 0:
        return 0.0
    return math.sqrt(float((audio * audio).mean()))
