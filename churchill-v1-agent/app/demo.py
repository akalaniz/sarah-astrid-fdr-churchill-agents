from __future__ import annotations

from app.core.config import load_settings
from app.core.openai_chat import FDROpenAIClient
from app.core.rag_context import format_sources
from app.core.sarah_response import generate_sarah_response, response_metadata
from app.persona.constitution import load_sarah_constitution
from app.persona.sarah_v2_system_prompt import build_sarah_system_prompt


DEMO_PROMPTS = [
    "Churchill, who are you?",
    "Darwin scares me.",
    "Iran tests a nuclear weapon. Give me COAs.",
]


def main() -> None:
    settings = load_settings()
    system_prompt = build_sarah_system_prompt()
    constitution = load_sarah_constitution()
    client = FDROpenAIClient(settings.openai_api_key)

    print("Churchill v1.0 demo")
    print(f"Model: {settings.sarah_model}")
    print(f"System prompt loaded: {len(system_prompt)} characters")
    print(f"Constitution loaded: {len(constitution.identity)} identity points")
    print(f"Vector store: {settings.vector_store_dir}")
    print()

    for index, prompt in enumerate(DEMO_PROMPTS, start=1):
        print("=" * 80)
        print(f"Demo prompt {index}: {prompt}")
        print("-" * 80)

        response = generate_sarah_response(
            user_input=prompt,
            settings=settings,
            client=client,
            history=[],
            system_prompt=system_prompt,
        )
        metadata = response_metadata(response)

        _safe_print(response.answer)
        print()
        print("Sources used:")
        _safe_print(format_sources(response.retrieval.results if response.retrieval else []))

        web_sources = metadata["web_sources"]
        if web_sources:
            print()
            print("Web/current sources:")
            for source in web_sources:
                _safe_print(f"- {source['title']} | {source['publisher']} | {source['date']} | {source['url']}")
        print()


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
