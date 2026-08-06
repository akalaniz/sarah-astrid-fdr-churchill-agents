from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass(frozen=True)
class DocumentSection:
    source_path: Path
    source_filename: str
    location: str
    text: str


def load_documents(source_dir: Path) -> list[DocumentSection]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    sections: list[DocumentSection] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        sections.extend(load_document(path))

    return [section for section in sections if section.text.strip()]


def load_document(path: Path) -> list[DocumentSection]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_text(path)
    if suffix == ".md":
        return _load_markdown(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _load_text(path: Path) -> list[DocumentSection]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [_section(path, "document", text)]


def _load_markdown(path: Path) -> list[DocumentSection]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    if not matches:
        return [_section(path, "document", text)]

    sections: list[DocumentSection] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        sections.append(_section(path, f"section: {heading}", text[start:end]))
    return sections


def _load_pdf(path: Path) -> list[DocumentSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Run: python -m pip install -e .") from exc

    reader = PdfReader(str(path))
    sections: list[DocumentSection] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        sections.append(_section(path, f"page {page_index}", text))
    return sections


def _load_docx(path: Path) -> list[DocumentSection]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX support requires python-docx. Run: python -m pip install -e .") from exc

    document = Document(str(path))
    sections: list[DocumentSection] = []
    current_heading = "document"
    current_parts: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style is not None else ""
        if style_name.lower().startswith("heading"):
            if current_parts:
                sections.append(_section(path, f"section: {current_heading}", "\n".join(current_parts)))
            current_heading = text
            current_parts = [text]
        else:
            current_parts.append(text)

    if current_parts:
        location = "document" if current_heading == "document" else f"section: {current_heading}"
        sections.append(_section(path, location, "\n".join(current_parts)))

    return sections


def _section(path: Path, location: str, text: str) -> DocumentSection:
    return DocumentSection(
        source_path=path.resolve(),
        source_filename=path.name,
        location=location,
        text=_normalize_text(text),
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()

