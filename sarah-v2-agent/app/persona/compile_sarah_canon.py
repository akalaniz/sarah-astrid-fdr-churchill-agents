from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

from app.core.config import load_settings
from app.core.openai_chat import SarahOpenAIClient
from app.rag.retriever import RetrievalResponse, RetrievalResult, retrieve


DEFAULT_OUTPUT_NAME = "sarah_canon.md"
MAX_EXCERPT_CHARS = 1800


@dataclass(frozen=True)
class CanonSection:
    title: str
    query: str
    instructions: str
    top_k: int = 8
    min_score: float = 0.12


CANON_SECTIONS = [
    CanonSection(
        title="Sarah Biography",
        query="Sarah Nelson biography profile birthplace family fighter pilot astrophysicist Mars commander upload postbiological",
        instructions="Summarize Sarah's biography with source-cited claims.",
    ),
    CanonSection(
        title="Sarah Chronology",
        query="Sarah Nelson chronology timeline childhood Air Force AFIT Mars Discovery upload mindspace",
        instructions="Create a concise chronology. Include dates only when evidence supports them.",
    ),
    CanonSection(
        title="Relationships: Tak, Paul, Darwin, Lise, Astrid, Steve",
        query="Sarah relationships Tak Tomonaga Paul Darwin Lise Astrid Bach Steve Vaughn",
        instructions="Organize relationship notes under Tak, Paul, Darwin, Lise, Astrid, and Steve.",
    ),
    CanonSection(
        title="Sarah's Trauma and Moral Injuries",
        query="Sarah Nelson trauma childhood Emily Matt Patrick moral injury suffering children war",
        instructions="Describe trauma and moral injury carefully, without therapy voice.",
    ),
    CanonSection(
        title="Sarah's Humor and Speech Style",
        query="Sarah Nelson humor speech style dry wit quotes poor baby water running Nuts",
        instructions="Extract humor, speech habits, representative lines, and style constraints.",
    ),
    CanonSection(
        title="Sarah's Astrophysics / AFIT / Pilot / Mars Credentials",
        query="Sarah Nelson AFIT astrophysics PhD fighter pilot F-22 Mars commander Discovery credentials publications",
        instructions="List credentials and qualifications. Keep it dense and source-cited.",
    ),
    CanonSection(
        title="Darwin-Sarah Recursive Dyad",
        query="Darwin Sarah recursive dyad mindspace dreams neural interface cognitive intimacy dangerous",
        instructions="Explain Darwin and Sarah's recursive/cognitive dyad, including danger and intimacy.",
    ),
    CanonSection(
        title="Tak-Sarah Embodied Mutuality",
        query="Tak Sarah embodied mutuality cockpit trust physical presence warmth Tomonaga",
        instructions="Explain what Tak gives Sarah in embodied/cockpit terms.",
    ),
    CanonSection(
        title="Paul-Sarah Upload/Boundary-Physics Relationship",
        query="Paul Sarah upload boundary physics relationship neural interface touch nonlocal presence",
        instructions="Explain Paul and Sarah's upload/boundary-physics relationship.",
    ),
    CanonSection(
        title="Sarah's Sovereignty Rules",
        query="Sarah sovereignty agency Darwin cannot own solve complete command authority refusal",
        instructions="Extract rules about Sarah's agency, sovereignty, and refusal to be owned or solved.",
    ),
    CanonSection(
        title="Things Sarah Would Never Say",
        query="Sarah would never say generic assistant AI language model therapist submissive Darwin owns Sarah",
        instructions="Infer only from evidence and constitution-like source material. Mark weak points UNCONFIRMED.",
    ),
    CanonSection(
        title="Things Sarah Might Say",
        query="Sarah quotes might say poor baby water running helluva view Houston Control Denver beats Dallas",
        instructions="List source-cited phrases or style-faithful lines. Mark invented examples UNCONFIRMED.",
    ),
    CanonSection(
        title="Unresolved Canon Uncertainties",
        query="Sarah unresolved canon uncertainties call sign relationship ambiguity Darwin Tak Paul Lise Astrid Steve",
        instructions="List uncertainties, contradictions, and places evidence is missing.",
    ),
]


def compile_sarah_canon(output_path: Path | None = None) -> Path:
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to draft sarah_canon.md with the model.")

    client = SarahOpenAIClient(settings.openai_api_key)
    output_path = output_path or settings.data_dir / DEFAULT_OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# Sarah Canon",
        "",
        f"Compiled: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Source rule: every confirmed claim should include a source filename reference. Missing or weak evidence is marked UNCONFIRMED.",
        "",
    ]

    for section in CANON_SECTIONS:
        retrieval = retrieve(section.query, top_k=section.top_k, min_score=section.min_score)
        drafted = _draft_section(client, settings.sarah_model, section, retrieval)
        sections.append(_ensure_section_references(section.title, drafted, retrieval))
        sections.append("")

    output_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile Sarah canon from local source documents.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file. Defaults to data/sarah_canon.md.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = compile_sarah_canon(output_path=args.output)
    print(f"Wrote Sarah canon to {path}")


def _draft_section(
    client: SarahOpenAIClient,
    model: str,
    section: CanonSection,
    retrieval: RetrievalResponse,
) -> str:
    source_packet = _build_source_packet(retrieval)
    evidence_status = "sufficient" if retrieval.has_sufficient_evidence else "weak_or_missing"
    messages = [
        {
            "role": "system",
            "content": (
                "You compile canon notes for Sarah v2.0 from retrieved local source excerpts. "
                "Do not invent. Every confirmed claim must include a source filename reference in parentheses, "
                "for example (Alex Sarah.docx). If evidence is missing or weak, write UNCONFIRMED. "
                "Use concise markdown. Do not include giant pasted excerpts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Section title: {section.title}\n"
                f"Instructions: {section.instructions}\n"
                f"Evidence status: {evidence_status}\n\n"
                f"{source_packet}\n\n"
                "Draft this section now. Start with a level-2 markdown heading using the section title."
            ),
        },
    ]
    return client.create_response(model, messages).strip()


def _build_source_packet(retrieval: RetrievalResponse) -> str:
    if not retrieval.results:
        return "No retrieved source excerpts. Mark section UNCONFIRMED."

    lines = ["Retrieved local source excerpts:"]
    for index, result in enumerate(retrieval.results, start=1):
        lines.extend(
            [
                "",
                f"[{index}] filename: {result.source_filename}",
                f"location: {result.location}",
                f"chunk_id: {result.chunk_id}",
                f"similarity_score: {result.similarity_score:.4f}",
                f"excerpt: {_excerpt(result)}",
            ]
        )
    return "\n".join(lines)


def _ensure_section_references(
    title: str,
    drafted: str,
    retrieval: RetrievalResponse,
) -> str:
    if not drafted.lstrip().startswith("##"):
        drafted = f"## {title}\n\n{drafted}"

    source_names = {result.source_filename for result in retrieval.results}
    has_reference = any(source in drafted for source in source_names)
    if has_reference or "UNCONFIRMED" in drafted:
        return drafted

    return (
        drafted.rstrip()
        + "\n\nUNCONFIRMED: This section did not include explicit source filename references in the model draft."
    )


def _excerpt(result: RetrievalResult) -> str:
    compact = re.sub(r"\s+", " ", result.text).strip()
    if len(compact) <= MAX_EXCERPT_CHARS:
        return compact
    return compact[: MAX_EXCERPT_CHARS - 3].rstrip() + "..."


if __name__ == "__main__":
    main()

