"""Loads and section-parses the institutional documents.

These are fictional documents for a wrapper entity ("Olist Inc.") per CLAUDE.md section 2 -
not real company data. They're stored as local markdown files, not Delta tables or a Databricks
Volume, since they aren't part of the Olist transactional dataset.

Parsing is by markdown ## header, giving clause/section-addressable text - this matches
CLAUDE.md's "section/clause-based chunking (not fixed token windows)" guidance for these docs,
implemented simply (no embeddings) for now. Real Vector Search with doc_type metadata tagging
is scoped to build order step 6, not built yet - search_policy_docs() in tools.py is a keyword
proxy in the meantime, same treatment Sentiment's search_reviews() got for the same reason.
"""
import re
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"

DOC_TYPES = {
    "company_registration.md": "registration",
    "hr_policy.md": "policy",
    "vendor_contract.md": "contract",
}


def _parse_sections(text: str) -> dict[str, str]:
    """Split on '## Header' lines into {header: body} pairs."""
    sections = {}
    current_header = None
    current_lines = []
    for line in text.splitlines():
        match = re.match(r"^##\s+(.+)$", line)
        if match:
            if current_header:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = match.group(1).strip()
            current_lines = []
        elif current_header:
            current_lines.append(line)
    if current_header:
        sections[current_header] = "\n".join(current_lines).strip()
    return sections


def load_document(filename: str) -> dict:
    """Returns {doc_type, filename, title, sections: {header: body}, raw_text}."""
    path = DOCUMENTS_DIR / filename
    text = path.read_text(encoding="utf-8")
    title_match = re.match(r"^#\s+(.+)$", text.splitlines()[0])
    return {
        "doc_type": DOC_TYPES.get(filename, "unknown"),
        "filename": filename,
        "title": title_match.group(1).strip() if title_match else filename,
        "sections": _parse_sections(text),
        "raw_text": text,
    }


def load_all_documents() -> list[dict]:
    return [load_document(f) for f in DOC_TYPES]


def find_section(doc: dict, topic: str) -> str | None:
    """Fuzzy match a topic string against a document's section headers (case-insensitive substring)."""
    topic_lower = topic.lower()
    for header, body in doc["sections"].items():
        if topic_lower in header.lower() or header.lower() in topic_lower:
            return header
    return None
