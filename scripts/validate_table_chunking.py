"""
Validation script for the table-header-preserving logic added to
structure_aware_chunk (_find_markdown_tables, _split_table_rows,
_split_section_preserving_table_headers).

Runs a set of synthetic edge cases plus every real table found in the
ingested corpus, and asserts invariants:
  - every table-continuation piece repeats the header + delimiter row
  - no table row is lost, duplicated, or reordered across pieces
  - no produced piece exceeds chunk_size tokens (except an unavoidable
    single-row overflow when one row alone is wider than the budget)
  - table rows never get merged into the same piece as surrounding
    prose, and rows from different tables never cross into each other

Kept out of src/ for the same reason as the other scripts/ utilities --
this is a dev-time check, not part of the pipeline.
"""

import sys
from pathlib import Path

import tiktoken

from src import config
from src.chunker import (
    TABLE_DELIMITER_ROW,
    _find_markdown_tables,
    _split_by_headers,
    _split_section_preserving_table_headers,
    _split_table_rows,
)
from src.ingest import load_corpus

enc = tiktoken.get_encoding("cl100k_base")
failures = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


def make_table(n_cols: int, n_rows: int, label: str = "") -> tuple[str, list[str]]:
    header = "| " + " | ".join(f"{label}Col{i}" for i in range(n_cols)) + " |"
    delimiter = "| " + " | ".join(["---"] * n_cols) + " |"
    body_rows = [
        "| " + " | ".join(f"{label}R{r}C{c}" for c in range(n_cols)) + " |"
        for r in range(n_rows)
    ]
    return f"{header}\n{delimiter}", body_rows


# --- Synthetic edge cases -------------------------------------------------

def test_small_table_stays_whole():
    header_text, body_rows = make_table(2, 3)
    table_text = header_text + "\n" + "\n".join(body_rows)
    pieces = _split_table_rows(header_text, body_rows, chunk_size=512, enc=enc)
    check(len(pieces) == 1, "small table: produces exactly one piece")
    check(pieces[0] == table_text, "small table: piece matches original table text")


def test_large_table_splits_with_repeated_header():
    header_text, body_rows = make_table(2, 20)
    header_tokens = len(enc.encode(header_text))
    row_tokens = len(enc.encode(body_rows[0]))
    chunk_size = header_tokens + row_tokens * 5  # forces ~5 rows per window

    pieces = _split_table_rows(header_text, body_rows, chunk_size, enc)
    check(len(pieces) > 1, "large table: splits into multiple pieces")

    reconstructed = []
    for piece in pieces:
        lines = piece.splitlines()
        check(lines[0] == header_text.splitlines()[0], "large table: piece starts with header row")
        check(TABLE_DELIMITER_ROW.match(lines[1].strip()) is not None,
              "large table: piece's second line is a delimiter row")
        reconstructed.extend(lines[2:])

    check(reconstructed == body_rows, "large table: all rows preserved in order, no loss/duplication")

    for piece in pieces[:-1]:
        tokens = len(enc.encode(piece))
        check(tokens <= chunk_size, f"large table: piece stays within budget ({tokens} <= {chunk_size})")


def test_table_with_surrounding_prose():
    header_text, body_rows = make_table(2, 30)
    table_text = header_text + "\n" + "\n".join(body_rows)
    section_text = (
        "Intro paragraph before the table explaining context.\n\n"
        + table_text
        + "\n\nClosing paragraph after the table.\n"
    )
    pieces = _split_section_preserving_table_headers(section_text, chunk_size=80, overlap=10, enc=enc)

    check(any("Intro paragraph" in p for p in pieces), "prose+table: intro paragraph preserved")
    check(any("Closing paragraph" in p for p in pieces), "prose+table: closing paragraph preserved")
    table_pieces = [p for p in pieces if "Col0" in p]
    check(len(table_pieces) > 1, "prose+table: table itself still gets split")
    check(
        all(not ("Intro paragraph" in p and "Col0" in p) for p in pieces),
        "prose+table: prose and table rows never share the same piece",
    )


def test_multiple_tables_in_one_section():
    header_a, rows_a = make_table(2, 12, label="A")
    header_b, rows_b = make_table(2, 12, label="B")
    section_text = (
        f"{header_a}\n" + "\n".join(rows_a)
        + "\n\nSome text between the two tables.\n\n"
        + f"{header_b}\n" + "\n".join(rows_b)
    )
    pieces = _split_section_preserving_table_headers(section_text, chunk_size=60, overlap=5, enc=enc)

    a_header_line = header_a.splitlines()[0]
    b_header_line = header_b.splitlines()[0]
    pieces_with_a_header = [p for p in pieces if p.splitlines()[0] == a_header_line]
    pieces_with_b_header = [p for p in pieces if p.splitlines()[0] == b_header_line]

    check(len(pieces_with_a_header) > 1, "multi-table: first table split into multiple pieces")
    check(len(pieces_with_b_header) > 1, "multi-table: second table split into multiple pieces")
    check(
        all("B" not in "".join(p.splitlines()[2:]) for p in pieces_with_a_header),
        "multi-table: first table's pieces contain no rows from the second table",
    )
    check(
        any("between the two tables" in p for p in pieces),
        "multi-table: text between tables preserved as its own piece",
    )


def test_empty_table_body_does_not_crash():
    section_text = "| A | B |\n| --- | --- |\n"
    tables = _find_markdown_tables(section_text)
    check(len(tables) == 1, "empty table: still detected as a table")
    header_line, end_line = tables[0]
    check(end_line - header_line == 2, "empty table: zero body rows found")


def test_malformed_table_not_detected():
    # header-like row followed by a row that isn't a valid delimiter
    section_text = "| A | B |\n| just text, no dashes |\n| more | rows |\n"
    tables = _find_markdown_tables(section_text)
    check(len(tables) == 0, "malformed table: not mistaken for a real table")


# --- Real corpus regression check -----------------------------------------

def test_real_corpus_tables():
    documents = load_corpus(config.DATA_RAW_DIR)
    checked_any = False

    for parser_name in ("regex", "mistune"):
        config.MARKDOWN_PARSER = parser_name
        for doc in documents:
            sections = _split_by_headers(doc.text)
            for section in sections:
                section_text = section["text"]
                tables = _find_markdown_tables(section_text)
                if not tables:
                    continue

                lines = section_text.splitlines()
                for header_line, end_line in tables:
                    header_text = "\n".join(lines[header_line:header_line + 2])
                    body_rows = lines[header_line + 2:end_line]
                    table_text = header_text + "\n" + "\n".join(body_rows)

                    if len(enc.encode(table_text)) <= config.CHUNK_SIZE:
                        continue  # not split, nothing to check

                    checked_any = True
                    pieces = _split_table_rows(header_text, body_rows, config.CHUNK_SIZE, enc)
                    label = f"{parser_name}/{Path(doc.source_path).name}"
                    header_lines = header_text.splitlines()

                    reconstructed = []
                    for piece in pieces:
                        piece_lines = piece.splitlines()
                        check(
                            piece_lines[:len(header_lines)] == header_lines,
                            f"{label}: table piece repeats header+delimiter verbatim",
                        )
                        reconstructed.extend(piece_lines[len(header_lines):])

                    check(
                        reconstructed == body_rows,
                        f"{label}: all {len(body_rows)} rows preserved in order (got {len(reconstructed)})",
                    )

                    for piece in pieces:
                        tokens = len(enc.encode(piece))
                        n_rows_in_piece = len(piece.splitlines()) - len(header_lines)
                        check(
                            tokens <= config.CHUNK_SIZE or n_rows_in_piece <= 1,
                            f"{label}: piece within token budget or unavoidable single-row overflow ({tokens} tokens)",
                        )

    check(checked_any, "real corpus: at least one oversized table was found and checked")


if __name__ == "__main__":
    original_parser = config.MARKDOWN_PARSER

    test_small_table_stays_whole()
    test_large_table_splits_with_repeated_header()
    test_table_with_surrounding_prose()
    test_multiple_tables_in_one_section()
    test_empty_table_body_does_not_crash()
    test_malformed_table_not_detected()
    test_real_corpus_tables()

    config.MARKDOWN_PARSER = original_parser

    print(f"\n{len(failures)} failure(s) out of all checks.")
    if failures:
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
