"""
One-time test-harness utility: run every question in tests/test_questions.md
through the live pipeline and write the retrieved chunks into the column
for the given chunking strategy.

I am keeping this utility script out of src/ folder, because this is not part 
of the RAG pipeline, rather this will call the RAG pipeline when I want to 
evaluate the search result from specific chunking strategy.
"""

from pathlib import Path
import re

from src import config, vectorstore
from src.pipeline import answer_question, build_index

strategy = config.CHUNKING_STRATEGY

# Matches a "|" that is NOT escaped with a backslash -- a real column
# boundary, not a literal pipe inside chunk text that _sanitize_cell()
# escaped as "\|". Plain str.split("|") treats every escaped pipe as a
# new column too, which silently misaligns the row (PDF table extraction
# puts a lot of literal "|" characters into chunk text).
_CELL_BOUNDARY = re.compile(r"(?<!\\)\|")

TEST_QUESTIONS_PATH = config.PROJECT_ROOT / "tests" / "test_questions.md"

# Column indices after splitting a table row on "|" -- index 0 and the last
# index are always the empty strings before/after the leading/trailing pipe.
QUESTION_COLUMN = 2

# Each strategy owns an "Actual answer" column at a fixed position in the
# single consolidated table (see tests/test_questions.md).
STRATEGY_ACTUAL_ANSWER_COLUMN = {
    "naive": 4,
    "structure_aware": 6,
    "recursive": 8,
}


def _split_row(line: str) -> list[str]:
    return _CELL_BOUNDARY.split(line)


def _sanitize_cell(text: str) -> str:
    """Flatten to one line and escape pipes so the text can't break the table."""
    return " ".join(text.split()).replace("|", "\\|")


def _format_actual_answer(result: dict) -> str:
    """Join the retrieved chunks (with source file + score) into one table cell."""
    parts = []
    for chunk in result["retrieved_chunks"]:
        source = Path(chunk["metadata"]["file_path"]).name
        chunk_index = chunk["metadata"]["chunk_index"]
        token_count = chunk["metadata"]["token_count"]
        text = _sanitize_cell(chunk["text"])
        parts.append(f"[File={source}, Chunk Index={chunk_index}, Token Count={token_count}, Score={chunk['score']:.3f}] {text}")
    return " <br><br> ".join(parts)


def fill_test_questions_table(strategy: str, top_k: int = 3) -> None:
    """
    Run every question in tests/test_questions.md through answer_question()
    and write the retrieved chunks into the given strategy's 'Actual answer'
    column, in place.
    """
    if strategy not in STRATEGY_ACTUAL_ANSWER_COLUMN:
        raise ValueError(
            f"Unknown strategy '{strategy}', expected one of {list(STRATEGY_ACTUAL_ANSWER_COLUMN)}"
        )
    actual_answer_column = STRATEGY_ACTUAL_ANSWER_COLUMN[strategy]

    lines = TEST_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()

    header_idx = next(i for i, line in enumerate(lines) if line.startswith("|"))
    row_idx = header_idx + 2  # skip the header row and the "|---|...|" separator row

    updated = 0
    failures = []

    while row_idx < len(lines) and lines[row_idx].startswith("|"):
        cells = _split_row(lines[row_idx])
        question = cells[QUESTION_COLUMN].strip()

        if question:
            print(f"Answering: {question}")
            try:
                result = answer_question(strategy=strategy, question=question, top_k=top_k)
                cells[actual_answer_column] = f" {_format_actual_answer(result)} "
                lines[row_idx] = "|".join(cells)
                updated += 1
            except Exception as e:
                print(f"  Failed: {e}")
                failures.append((question, str(e)))

        row_idx += 1

    TEST_QUESTIONS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nUpdated {updated} row(s) for strategy '{strategy}' in {TEST_QUESTIONS_PATH}")

    if failures:
        print(f"\n{len(failures)} question(s) failed:")
        for question, error in failures:
            print(f"  - {question}: {error}")


if __name__ == "__main__":

    try:
        # One-time building the vector store index 
        build_index(strategy=strategy)

        fill_test_questions_table(strategy=strategy, top_k=3)

    except Exception as e:
        print(f"Error while answering question: {e}")

    finally:
        # Always close the Weaviate connection, even if build_index/answer_question
        # raised -- otherwise a failed run leaks the open client connection.
        vectorstore.close_client()