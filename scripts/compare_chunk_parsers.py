"""
One-off debug utility: run structure_aware_chunk on a single document
under both header-split strategies (plain regex vs. mistune AST) and
write the full output for each to its own file, so you can diff them
side by side (e.g. VS Code's "Compare Active File With...").

Kept out of src/ for the same reason as run_test_questions.py -- this is
a manual inspection tool, not part of the pipeline itself.
"""

from src import config, chunker
from src.chunker import Chunk
from src.ingest import load_corpus

TARGET_FILENAME = "rural-health-transformation-50-state-spotlights.pdf"
OUTPUT_DIR = config.PROJECT_ROOT / "debug_output"


def load_target_document():
    documents = load_corpus(config.DATA_RAW_DIR)
    for doc in documents:
        if doc.source_path.endswith(TARGET_FILENAME):
            return doc
    raise FileNotFoundError(f"{TARGET_FILENAME} not found under {config.DATA_RAW_DIR}")


def format_chunks(chunks: list[Chunk]) -> str:
    lines = [f"Total chunks: {len(chunks)}\n"]
    for c in chunks:
        lines.append(
            f"[{c.chunk_id}] heading_path={c.metadata['heading_path']!r} "
            f"token_count={c.metadata['token_count']}"
        )
        lines.append(c.text)
        lines.append("-" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    doc = load_target_document()
    print(f"Loaded '{TARGET_FILENAME}' as doc_id={doc.doc_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for parser_name in ("regex", "mistune"):
        config.MARKDOWN_PARSER = parser_name
        chunks = chunker.structure_aware_chunk(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        results[parser_name] = chunks

        output_path = OUTPUT_DIR / f"structure_aware_{parser_name}.txt"
        output_path.write_text(format_chunks(chunks), encoding="utf-8")
        print(f"{parser_name}: {len(chunks)} chunks -> {output_path}")

    regex_paths = [c.metadata["heading_path"] for c in results["regex"]]
    mistune_paths = [c.metadata["heading_path"] for c in results["mistune"]]

    print(f"\nChunk count -- regex: {len(regex_paths)}, mistune: {len(mistune_paths)}")

    # Equal chunk counts don't guarantee identical section boundaries, so
    # point out exactly where the two heading_path sequences diverge.
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(regex_paths, mistune_paths)) if a != b),
        None,
    )
    if first_diff is not None:
        print(f"First differing heading_path at chunk index {first_diff}:")
        print(f"  regex:   {regex_paths[first_diff]!r}")
        print(f"  mistune: {mistune_paths[first_diff]!r}")
    elif len(regex_paths) != len(mistune_paths):
        print("heading_path sequences match up to the shorter list; lengths differ.")
    else:
        print("heading_path sequences are identical across both parsers.")
