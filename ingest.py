"""
Ingestion: load documents from ./data, chunk them, embed each chunk,
and write the vectors into a pgvector-backed table via LlamaIndex.

Two source types get specialized, format-aware parsing instead of the
generic directory reader (see parsers.py for why each needs it):

  - PPDM error/event catalog PDFs -- one TextNode per catalog row
    (error_code/severity/category as metadata) instead of chunking the
    flattened table text arbitrarily.
  - PPDM SOP docx files, one per error code -- one TextNode per Markdown
    section (error_code/section as metadata) instead of losing document
    structure to a generic splitter.

Any .pdf/.docx that doesn't fit either shape (parsers.py raises) falls
back to the normal SimpleDirectoryReader + chunker, same as every other
file type in ./data.

Requires poppler-utils (`pdftotext`) and `pandoc` on PATH -- not
pip-installable, see README.md.

Run: python ingest.py
"""

from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from common import configure_llamaindex_settings, get_vector_store
from parsers import parse_error_reference, parse_sop

configure_llamaindex_settings()
Settings.chunk_size = 512  # 1024 default chunk size can be too large for multi-hop questions.
Settings.chunk_overlap = round(Settings.chunk_size * 0.15)

vector_store = get_vector_store()

DATA_DIR = Path("./data")


def _load_special_nodes() -> tuple:
    """Parses every .pdf/.docx directly under DATA_DIR that fits the PPDM
    catalog/SOP shape into pre-chunked TextNodes. Returns (nodes, handled)
    where `handled` is the set of file paths parsed this way, so the
    generic loader below can skip them."""
    nodes = []
    handled = set()

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            pdf_nodes = parse_error_reference(str(pdf_path))
        except (ValueError, RuntimeError) as exc:
            print(f"  {pdf_path.name}: not an event-catalog PDF ({exc}) — "
                  f"falling back to generic ingestion", flush=True)
            continue
        print(f"  {pdf_path.name}: {len(pdf_nodes)} catalog row(s) parsed", flush=True)
        nodes.extend(pdf_nodes)
        handled.add(pdf_path)

    for docx_path in sorted(DATA_DIR.glob("*.docx")):
        try:
            sop_nodes = parse_sop(str(docx_path))
        except (ValueError, RuntimeError) as exc:
            print(f"  {docx_path.name}: not a parseable SOP ({exc}) — "
                  f"falling back to generic ingestion", flush=True)
            continue
        error_code = sop_nodes[0].metadata["error_code"] if sop_nodes else "?"
        print(f"  {docx_path.name}: {len(sop_nodes)} section(s) parsed "
              f"(error_code={error_code})", flush=True)
        nodes.extend(sop_nodes)
        handled.add(docx_path)

    return nodes, handled


def main() -> None:
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise SystemExit("No documents found in ./data — add a file and re-run.")

    print("Checking ./data for PPDM catalog/SOP files...", flush=True)
    special_nodes, handled_paths = _load_special_nodes()

    generic_paths = [
        str(p) for p in sorted(DATA_DIR.glob("*"))
        if p.is_file() and p not in handled_paths
    ]
    generic_nodes = []
    if generic_paths:
        print(f"Loading {len(generic_paths)} remaining document(s) from ./data...", flush=True)
        documents = SimpleDirectoryReader(input_files=generic_paths).load_data()
        print(
            f"Chunking (size={Settings.chunk_size}, overlap={Settings.chunk_overlap})...",
            flush=True,
        )
        splitter = SentenceSplitter(
            chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
        )
        generic_nodes = splitter.get_nodes_from_documents(documents)

    all_nodes = special_nodes + generic_nodes
    if not all_nodes:
        raise SystemExit("No nodes produced from ./data — nothing to ingest.")

    print(f"Embedding {len(all_nodes)} node(s) into pgvector — progress below:", flush=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Nodes are passed pre-built (not raw Documents), since the special
    # parsers already did the chunking -- VectorStoreIndex only needs to
    # embed and write them.
    VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)

    print(
        f"Done. Ingested {len(special_nodes)} catalog/SOP node(s) + "
        f"{len(generic_nodes)} generic node(s) into pgvector table 'rag_chunks'.",
        flush=True,
    )


if __name__ == "__main__":
    main()