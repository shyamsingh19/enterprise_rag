#!/usr/bin/env python3
"""One-off indexing script: run this upfront to load data/documents/ into Chroma.

Usage:
    python scripts/ingest.py [directory]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingestion import index_directory  # noqa: E402


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/documents")
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    results = index_directory(directory)
    if not results:
        print(f"No supported documents (.txt, .md, .pdf) found in {directory}")
        return

    for name, chunk_count in results.items():
        print(f"  {name}: {chunk_count} chunks")
    print(f"Indexed {len(results)} file(s), {sum(results.values())} chunks total.")


if __name__ == "__main__":
    main()
