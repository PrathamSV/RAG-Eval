"""
Parsers for the two PPDM RAG source documents.

Both source files are genuine binary Office documents (confirmed via
`file`: PDF 1.6, and a real OOXML docx) -- NOT pre-flattened plain text.
Each parser does a real extraction pass first, then the same
format-aware row/section logic as before:

  - ppdm1920-events-code_1.pdf -> the table has no vertical ruling lines,
    so pdfplumber's table detection collapses every row into a single
    unstructured cell (verified). Extraction goes through text instead:
    `pdftotext -raw` (content-stream order, NOT `-layout`'s reading-order
    reconstruction) is what makes "Severity + Category always immediately
    precede the next Message ID" hold true in the extracted text --
    verified against the real PDF: 2,302 unique rows, 0 duplicates, 100%
    severity/category extraction. Plain `pdftotext` and `-layout` both
    scramble the column order (403 / 131 rows respectively) -- `-raw`
    specifically matters here.

  - PPDM_ABA0008_Sample_SOP.docx -> converted with `pandoc -t gfm` (NOT
    pandoc's default markdown writer, which emits grid tables instead of
    pipe tables) into real Markdown (#/## headers, **bold**, pipe
    tables), then parsed with LlamaIndex's MarkdownNodeParser, which
    chunks per section and preserves header_path as metadata.

Both require external CLI tools on PATH: poppler-utils (`pdftotext`) and
`pandoc`. Neither is pip-installable -- install via apt/brew (see README).

Drop into ingest.py in place of SimpleDirectoryReader for these two files.
"""

import re
import subprocess
from collections import Counter
from pathlib import Path
import pdfplumber

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode

# Categories observed in the catalog (25 distinct values). Longest-first
# so multi-word categories match before their single-word prefixes
# (e.g. "Protection Policy" before bare "Protection").
_CATEGORIES = [
    "App Host Configuration", "Cndm Server Dr", "Nas Server Dr", "Protection Copy",
    "Protection Infrastructure", "Protection Policy", "Protection Source",
    "Restore Plan", "Self Service", "Server Dr", "Push Update", "Cloud Tier",
    "Instant Access", "Compliance", "Discover", "License", "Migrate",
    "Protection", "Replication", "Reporting", "Restore", "Security",
    "System", "Export Log", "Agent",
]
_CAT_ALT = "|".join(re.escape(c) for c in sorted(_CATEGORIES, key=len, reverse=True))

# Error-code shape shared by the catalog table and SOP filenames/titles,
# e.g. ARO0506, ABA0008, ARSGHV0001.
_ERROR_CODE = r"[A-Z][A-Z]{1,9}\d{3,5}"

# Loose candidate pattern for pulling a code-shaped token out of *free
# text* (a user's question, a pasted log line) -- unlike _ERROR_CODE
# above, this tolerates the punctuation/whitespace people actually type
# or paste between the letter and digit runs: "ABA-0008", "aba 0008",
# "ABA_0008", lowercase, extra spaces. normalize_error_code() strips that
# separator and re-checks the result against _ERROR_CODE's canonical
# shape, so a false-positive candidate (e.g. "in 2026" -> "IN2026" also
# happens to fit the letters+digits shape) still has to actually exist in
# the ingested catalog to do anything -- callers treat a DB miss as "not
# a real code" and fall back to normal retrieval, not as an error.
_CODE_CANDIDATE = re.compile(r"\b([A-Za-z]{2,10})[\s\-_]{0,2}(\d{3,5})\b")

# Bootstrap pass: same anchor shape as the real row-boundary regex, but
# permissive on the category text itself -- whatever short capitalized
# phrase sits between a severity word and the next ID-shaped token.
# This makes the whitelist self-updating instead of a hand-typed list
# that silently goes stale when the catalog adds a category (as it
# already has: "Anomaly Detection" and "Protection Rule" aren't in the
# old hardcoded list and were getting merged into the previous row).
_CATEGORY_BOOTSTRAP = re.compile(
    rf'(?:Critical|Warning|Informational)\s+'
    rf'((?:[A-Z][a-zA-Z]*\s?){{1,3}}?)'
    rf'(?={_ERROR_CODE}(?=\s))'
)

# Anchor a row boundary on the actual table structure (Severity + Category
# always precede the next Message ID) rather than on whitespace patterns,
# which vary between page-boundary rows and mid-page rows.
_ROW_BOUNDARY = re.compile(
    rf'(?:Critical|Warning|Informational)\s+(?:{_CAT_ALT})\s*'
    rf'({_ERROR_CODE})(?=\s)'
)

_FURNITURE = re.compile(
    r'PowerProtect Data Manager 19\.20 Message Catalog|'
    r'Copyright © 2025 Dell Inc\..*?respective owners\.|'
    r'Dell PowerProtect Data Manager Messages Catalog|'
    r'Message ID Message Details Recommended Action Severity Category',
    re.DOTALL,
)
_SEVERITY_TAIL = re.compile(r'\b(Critical|Warning|Informational)\s+([A-Za-z][A-Za-z /]*?)\s*$')
_TRAILING_PAGE_NUMS = re.compile(r'(\s+\d{1,3})+$')


def normalize_error_code(text: str) -> str | None:
    """Best-effort extraction + normalization of a PPDM error code from
    free text (a user's question, a pasted log line, etc.).

    Handles the formatting variants people actually type or paste --
    "ABA-0008", "aba 0008", "ABA_0008", lowercase, extra whitespace -- by
    stripping the separator between the letter and digit runs and
    uppercasing, then checking the normalized result against the
    canonical code shape (_ERROR_CODE: 2-10 letters + 3-5 digits, the
    same shape the catalog PDF and SOP filenames/titles use).

    Returns the normalized code (e.g. "ABA0008") if `text` contains
    something code-shaped, else None. Does NOT verify the code actually
    exists in the ingested catalog -- that's a DB lookup
    (fetch_nodes_by_error_code in common.py), not this function's job.
    Deliberately intersects two independent regexes (a loose free-text
    candidate here, the canonical catalog shape) rather than one combined
    pattern, so a stray digit run in ordinary prose ("in 2026") can match
    the loose candidate without silently becoming "correct" -- it still
    has to pass the same shape check a real catalog code does, and even
    then a DB miss just falls back to normal retrieval upstream.
    """
    match = _CODE_CANDIDATE.search(text)
    if not match:
        return None
    candidate = f"{match.group(1)}{match.group(2)}".upper()
    return candidate if re.fullmatch(_ERROR_CODE, candidate) else None


def _run(cmd: list) -> str:
    """Run a CLI tool and return stdout, with a clear error if it's missing
    (rather than a raw FileNotFoundError pointing at subprocess internals)."""
    try:
        # encoding="utf-8" is required explicitly: with text=True and no
        # encoding, subprocess decodes using the platform's default locale
        # encoding (cp1252 on Windows), not UTF-8. pandoc's gfm output is
        # UTF-8 (e.g. curly quotes, em dashes, non-breaking spaces pulled
        # from the docx), so on Windows that mismatch raises
        # UnicodeDecodeError on the first non-ASCII byte. errors="replace"
        # keeps a stray bad byte from crashing ingestion outright, swapping
        # it for U+FFFD instead of failing the whole run.
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required CLI tool '{cmd[0]}' not found on PATH. Install poppler-utils "
            f"(pdftotext) and pandoc -- see README.md."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{cmd[0]} failed on {cmd[-1]}: {exc.stderr}") from exc
    return result.stdout


def _discover_categories(raw: str) -> list:
    """All distinct category strings actually used in this document,
    harvested from the text itself rather than maintained by hand."""
    return sorted({m.group(1).strip() for m in _CATEGORY_BOOTSTRAP.finditer(raw)},
                  key=len, reverse=True)


def parse_error_reference(path: str) -> list:
    """One TextNode per catalog row, with error_code/severity/category metadata.

    Text pulled via pdfplumber with use_text_flow=True -- this reads
    words in PDF content-stream order rather than reconstructing reading
    order by position, which is what makes "Severity + Category always
    immediately precede the next Message ID" hold true in the extracted
    text. (Default pdfplumber extraction, and pdftotext without -raw,
    both reorder by position and scramble this table's column layout.)
    """
    with pdfplumber.open(path) as pdf:
        raw = "\n".join(
            " ".join(w["text"] for w in page.extract_words(use_text_flow=True, keep_blank_chars=False))
            for page in pdf.pages
        )

    categories = _discover_categories(raw)
    cat_alt = "|".join(re.escape(c) for c in categories)
    row_boundary = re.compile(
        rf'(?:Critical|Warning|Informational)\s+(?:{cat_alt})\s*'
        rf'({_ERROR_CODE})(?=\s)'
    )

    matches = list(row_boundary.finditer(raw))
    if not matches:
        raise ValueError(f"No rows detected in {path} — check the row-boundary regex")

    nodes = []
    for i, m in enumerate(matches):
        code = m.group(1)
        start = m.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(raw)

        content = _FURNITURE.sub("", raw[start:end])
        content = re.sub(r"\s+", " ", content).strip()
        content = _TRAILING_PAGE_NUMS.sub("", content)

        sev_match = _SEVERITY_TAIL.search(content)
        severity = sev_match.group(1) if sev_match else None
        category = sev_match.group(2).strip() if sev_match else None

        nodes.append(TextNode(
            text=content,
            metadata={
                "error_code": code,
                "doc_type": "reference",
                "severity": severity,
                "category": category,
                "source": Path(path).name,
            },
        ))

    codes = [n.metadata["error_code"] for n in nodes]
    dupes = {c: n for c, n in Counter(codes).items() if n > 1}
    if dupes:
        raise ValueError(f"Duplicate error codes detected (regex likely mis-split a row): {dupes}")

    return nodes


def _guess_error_code(path: str, markdown_text: str) -> str | None:
    """Best-effort error code for an SOP file that isn't told its code
    explicitly. Tries the filename first (SOPs are conventionally named
    like `PPDM_ABA0008_Sample_SOP.docx`), then falls back to the first
    error-code-shaped token in the document body (SOP titles conventionally
    read "PPDM Error ABA0008 - ...")."""
    filename_match = re.search(_ERROR_CODE, Path(path).stem)
    if filename_match:
        return filename_match.group(0)
    body_match = re.search(_ERROR_CODE, markdown_text[:500])
    return body_match.group(0) if body_match else None


def parse_sop(path: str, error_code: str | None = None) -> list:
    """One TextNode per Markdown section, tagged with the error_code it documents.

    `path` is a real .docx. Converted with `pandoc -t gfm` (GitHub-Flavored
    Markdown -- pandoc's default markdown writer emits grid tables instead
    of pipe tables, which MarkdownNodeParser doesn't parse as tables) into
    real Markdown, then parsed with LlamaIndex's MarkdownNodeParser.

    If `error_code` isn't passed, it's guessed from the filename or the
    document title (see `_guess_error_code`) -- useful for ingesting a
    whole folder of SOPs, one per error code, rather than a single known
    file.
    """
    markdown_text = _run(["pandoc", "-t", "gfm", path])

    if error_code is None:
        error_code = _guess_error_code(path, markdown_text)
        if error_code is None:
            raise ValueError(
                f"Couldn't determine which error code {path} documents -- "
                f"pass error_code explicitly, or name the file/title with "
                f"the code (e.g. PPDM_ABA0008_Sample_SOP.docx)."
            )

    raw_nodes = MarkdownNodeParser().get_nodes_from_documents(
        [Document(text=markdown_text)]
    )

    return [
        TextNode(
            text=n.text,
            metadata={
                "error_code": error_code,
                "doc_type": "sop",
                "section": n.metadata.get("header_path", "/"),
                "source": Path(path).name,
            },
        )
        for n in raw_nodes
    ]


if __name__ == "__main__":
    ref_nodes = parse_error_reference("/mnt/project/ppdm1920-events-code_1.pdf")
    sop_nodes = parse_sop("/mnt/project/PPDM_ABA0008_Sample_SOP.docx")

    print(f"Reference table: {len(ref_nodes)} rows parsed")
    print(f"SOP: {len(sop_nodes)} sections parsed (error_code={sop_nodes[0].metadata['error_code']})")

    aba0008 = next(n for n in ref_nodes if n.metadata["error_code"] == "ABA0008")
    print("\n--- ABA0008 reference row ---")
    print(aba0008.text)
    print(aba0008.metadata)