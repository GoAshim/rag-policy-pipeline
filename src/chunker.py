"""
Chunking: split RawDocuments into smaller Chunk objects for embedding
and retrieval.
"""

import re
from dataclasses import dataclass
from src import config
from src.ingest import RawDocument
import tiktoken
import mistune

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str     # which RawDocument this came from
    text: str
    metadata: dict  # inherit/extend the parent doc's metadata


def naive_fixed_size(doc: RawDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """
    Week 1 baseline: split text into fixed-size token windows with some
    overlap.
    """

    if not doc.text.strip():
        return []

    if (overlap < 0 ) or (chunk_size <= overlap):
        raise ValueError("overlap must be a positive number and less than chunk_size")
    
    # Set the tokenizer model and load it locally
    tokenizer_model = "cl100k_base"
    enc = tiktoken.get_encoding(tokenizer_model)

    tokens = enc.encode(doc.text)
    total_tokens = len(tokens)
    step = chunk_size - overlap

    chunks = []
    chunk_idx = 0
    for start_idx in range(0, total_tokens, step):

        # Find the start and end token position of each chunk
        end_idx = min(start_idx + chunk_size, total_tokens)
        # Split the tokens into chunks
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode token sequence back to text
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
            doc_id=doc.doc_id,
            text=chunk_text,
            metadata={
                "source_path": doc.source_path,
                "chunk_index": chunk_idx,
                "start_token": start_idx,
                "end_token": end_idx,
                "token_count": len(chunk_tokens),
                **doc.metadata,  # Inherit title, page counts, etc. from the parent doc
            },
        ))
        chunk_idx += 1

        if end_idx == total_tokens:
            break

    return chunks


# _recursive_token_split is a helper function used by both 
# recursive_overlap_chunk and structure_aware_chunk. 
def _recursive_token_split(
    text: str,
    chunk_size: int,
    overlap: int,
    enc,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split `text` on a priority list of separators (paragraphs,
    then lines, then sentences, then words, then raw characters), only
    descending into a piece when it's still too big, then pack the
    resulting pieces into chunk_size-token windows with `overlap` tokens
    of trailing context carried into the next window.

    It returns plain chunk text strings.
    """

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def token_len(s: str) -> int:
        return len(enc.encode(s))

    def split_on_separator(s: str, sep: str) -> list[str]:
        if not sep:
            return list(s)
        parts = s.split(sep)
        return [p + sep for p in parts[:-1]] + parts[-1:]

    def split_text(s: str, seps: list[str]) -> list[str]:
        if not seps:
            return [s]

        sep, remaining_seps = seps[0], seps[1:]
        pieces = []
        for piece in split_on_separator(s, sep):
            if not piece:
                continue
            if token_len(piece) <= chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(split_text(piece, remaining_seps))
        return pieces

    def merge_pieces(pieces: list[str]) -> list[str]:
        chunk_texts = []
        current = ""
        current_len = 0

        for piece in pieces:
            piece_len = token_len(piece)

            if current and current_len + piece_len > chunk_size:
                chunk_texts.append(current)
                tail_tokens = enc.encode(current)[-overlap:] if overlap else []
                current = enc.decode(tail_tokens)
                current_len = len(tail_tokens)

            current += piece
            current_len += piece_len

        if current.strip():
            chunk_texts.append(current)

        return chunk_texts

    pieces = split_text(text, separators)
    return merge_pieces(pieces)


def recursive_overlap_chunk(
    doc: RawDocument,
    chunk_size: int,
    overlap: int,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """
    Recursively split on a priority list of separators (paragraphs, 
    then lines, then sentences, then words, then raw characters),
    only descending into a piece when it's still too big, then 
    pack the resulting pieces into chunk_size-token windows with
    `overlap` tokens of trailing context into the next window.
    """

    if not doc.text.strip():
        return []

    if (overlap < 0) or (chunk_size <= overlap):
        raise ValueError("overlap must be a positive number and less than chunk_size")

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    tokenizer_model = "cl100k_base"
    enc = tiktoken.get_encoding(tokenizer_model)

    chunk_texts = _recursive_token_split(doc.text, chunk_size, overlap, enc, separators)

    chunks = []
    for chunk_idx, chunk_text in enumerate(chunk_texts):
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
            doc_id=doc.doc_id,
            text=chunk_text,
            metadata={
                "source_path": doc.source_path,
                "chunk_index": chunk_idx,
                "token_count": len(enc.encode(chunk_text)),
                **doc.metadata,
            },
        ))

    return chunks


HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


# _split_by_headers is a helper function for structure_aware_chunk
def _split_by_headers(text: str) -> list[dict]:
    """
    Split markdown text into sections at ATX-style headers. Each section
    runs from one header to the next header of equal-or-higher level (or
    end of text), and carries the stack of ancestor headings it's nested
    under.

    Dispatches to a real markdown-AST parser (mistune) or a plain regex
    scan depending on config.MARKDOWN_PARSER. Refer each helper function
    for the tradeoff.
    """

    if config.MARKDOWN_PARSER == "mistune":
        return _split_by_headers_mistune(text)

    return _split_by_headers_regex(text)


def _split_by_headers_regex(text: str) -> list[dict]:
    """
    Scan for ATX headers with a plain regex. Cheap and preserves section
    bodies verbatim (exact original formatting, tables included), but
    can't tell a real header from a '#' that happens to start a line
    inside a fenced code block, blockquote, etc.
    """

    matches = list(HEADER_PATTERN.finditer(text))

    if not matches:
        return [{"heading_path": [], "text": text}]

    sections = []
    heading_stack = []  # list of (level, title)

    # Any text before the first header gets its own headerless section
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append({"heading_path": [], "text": preamble})

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Pop siblings/descendants of the same-or-deeper level, then push
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        sections.append({
            "heading_path": [t for _, t in heading_stack],
            "text": section_text,
        })

    return sections


def _render_inline(nodes: list[dict]) -> str:
    """
    Flatten a list of mistune inline AST nodes back into markdown text,
    so the heading titles and section bodies keep the emphasis markers 
    that the callers already expect.
    """

    parts = []
    for node in nodes:
        node_type = node.get("type")
        children = node.get("children")

        if node_type == "text":
            parts.append(node.get("raw", ""))
        elif node_type == "codespan":
            parts.append(f"`{node.get('raw', '')}`")
        elif node_type == "strong":
            parts.append(f"**{_render_inline(children or [])}**")
        elif node_type == "emphasis":
            parts.append(f"*{_render_inline(children or [])}*")
        elif node_type in ("linebreak", "softbreak"):
            parts.append("\n")
        elif node_type == "link":
            url = node.get("attrs", {}).get("url", "")
            label = _render_inline(children or [])
            parts.append(f"[{label}]({url})" if url else label)
        elif node_type == "image":
            url = node.get("attrs", {}).get("url", "")
            alt = _render_inline(children or [])
            parts.append(f"![{alt}]({url})" if url else alt)
        elif children is not None:
            parts.append(_render_inline(children))
        else:
            parts.append(node.get("raw", ""))

    return "".join(parts)


def _render_block(token: dict) -> str:
    """
    Render one mistune block-level AST node back into plain
    markdown text. Only used by the mistune path. The AST carries no
    character offsets, so section bodies have to be reconstructed from
    the tree instead of sliced out of the original string.
    """

    token_type = token.get("type")
    children = token.get("children", [])

    if token_type in ("paragraph", "block_text"):
        return _render_inline(children)
    if token_type == "block_code":
        info = token.get("attrs", {}).get("info", "") or ""
        return f"```{info}\n{token.get('raw', '')}```"
    if token_type == "block_quote":
        inner = "\n".join(_render_block(c) for c in children)
        return "\n".join(f"> {line}" for line in inner.splitlines())
    if token_type == "thematic_break":
        return "---"
    if token_type == "block_html":
        return token.get("raw", "")
    if token_type == "blank_line":
        return ""
    if token_type == "list":
        ordered = token.get("attrs", {}).get("ordered", False)
        start = token.get("attrs", {}).get("start") or 1
        lines = []
        for i, item in enumerate(children):
            prefix = f"{start + i}. " if ordered else "- "
            rendered_children = [_render_block(c) for c in item.get("children", [])]
            item_text = "\n".join(r for r in rendered_children if r)
            lines.append(prefix + item_text.replace("\n", "\n  "))
        return "\n".join(lines)
    if token_type == "table":
        rows = []
        for section in children:
            if section.get("type") == "table_head":
                cells = [_render_inline(c.get("children", [])) for c in section.get("children", [])]
                rows.append("| " + " | ".join(cells) + " |")
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            elif section.get("type") == "table_body":
                for row in section.get("children", []):
                    cells = [_render_inline(c.get("children", [])) for c in row.get("children", [])]
                    rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    # Unrecognized block type: recurse into children if there are any,
    # otherwise fall back to whatever raw text the token carries.
    if children:
        return "\n".join(_render_block(c) for c in children)
    return token.get("raw", "")


def _render_tokens(tokens: list[dict]) -> str:
    rendered = [_render_block(t) for t in tokens]
    return "\n\n".join(r for r in rendered if r)


def _split_by_headers_mistune(text: str) -> list[dict]:
    """
    Walk mistune's block AST to find genuine ATX headings -- so a '#'
    inside a fenced code block or blockquote is never mistaken for a
    real header, unlike the regex path. The AST carries no character
    offsets, so each section's body is reconstructed by
    re-rendering its non-heading tokens rather than sliced verbatim out
    of the source text.
    """

    parser = mistune.create_markdown(renderer=None, plugins=["table"])
    tokens = parser(text)

    if not any(t.get("type") == "heading" for t in tokens):
        return [{"heading_path": [], "text": text}]

    sections = []
    heading_stack = []  # list of (level, title)
    current_heading = None  # (level, title), or None for the preamble
    body_tokens = []

    def flush():
        body_text = _render_tokens(body_tokens).strip()
        if current_heading is None:
            if body_text:
                sections.append({"heading_path": [], "text": body_text})
            return

        level, title = current_heading
        heading_line = f"{'#' * level} {title}"
        section_text = f"{heading_line}\n\n{body_text}" if body_text else heading_line
        sections.append({
            "heading_path": [t for _, t in heading_stack],
            "text": section_text,
        })

    for token in tokens:
        if token.get("type") == "heading":
            flush()

            level = token["attrs"]["level"]
            title = _render_inline(token.get("children", [])).strip()

            # Pop siblings/descendants of the same-or-deeper level, then push
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            current_heading = (level, title)
            body_tokens = []
        else:
            body_tokens.append(token)

    flush()
    return sections


TABLE_DELIMITER_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _find_markdown_tables(text: str) -> list[tuple[int, int]]:
    """
    Locate GFM pipe tables in `text`. Returns (header_line, end_line)
    index pairs into text.splitlines() -- header_line is the header row,
    end_line is one past the last contiguous row that follows the
    delimiter row (e.g. '| --- | --- |').
    """

    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines) - 1:
        if "|" in lines[i] and TABLE_DELIMITER_ROW.match(lines[i + 1].strip()):
            header_line = i
            end_line = i + 2
            while end_line < len(lines) and "|" in lines[end_line] and lines[end_line].strip():
                end_line += 1
            tables.append((header_line, end_line))
            i = end_line
        else:
            i += 1
    return tables


def _split_table_rows(header_text: str, body_rows: list[str], chunk_size: int, enc) -> list[str]:
    """
    Pack table body rows into chunk_size-token windows, re-adding the
    header (+ delimiter) row to the top of every window past the first
    so a table that has to be split still tells you what each column
    means.
    """

    header_tokens = len(enc.encode(header_text))
    budget = max(chunk_size - header_tokens, 1)

    windows = []
    current_rows = []
    current_len = 0

    for row in body_rows:
        row_len = len(enc.encode(row))
        if current_rows and current_len + row_len > budget:
            windows.append(current_rows)
            current_rows = []
            current_len = 0
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        windows.append(current_rows)

    return [header_text + "\n" + "\n".join(rows) for rows in windows]


def _split_section_preserving_table_headers(text: str, chunk_size: int, overlap: int, enc) -> list[str]:
    """
    Split an over-budget section the same way _recursive_token_split
    does, except any markdown table gets carved out and packed
    separately so its header row is repeated at the top of every piece
    the table gets split into.
    """

    lines = text.splitlines()
    tables = _find_markdown_tables(text)

    if not tables:
        return _recursive_token_split(text, chunk_size, overlap, enc)

    sub_texts = []
    cursor = 0

    for header_line, end_line in tables:
        if header_line > cursor:
            pre_text = "\n".join(lines[cursor:header_line])
            if pre_text.strip():
                sub_texts.extend(_recursive_token_split(pre_text, chunk_size, overlap, enc))

        header_text = "\n".join(lines[header_line:header_line + 2])
        body_rows = lines[header_line + 2:end_line]
        table_text = header_text + "\n" + "\n".join(body_rows)

        if len(enc.encode(table_text)) <= chunk_size:
            sub_texts.append(table_text)
        else:
            sub_texts.extend(_split_table_rows(header_text, body_rows, chunk_size, enc))

        cursor = end_line

    if cursor < len(lines):
        tail_text = "\n".join(lines[cursor:])
        if tail_text.strip():
            sub_texts.extend(_recursive_token_split(tail_text, chunk_size, overlap, enc))

    return sub_texts


def structure_aware_chunk(
    doc: RawDocument,
    chunk_size: int = 512,
    overlap: int = 75,
) -> list[Chunk]:
    """
    Split on document structure (headers, sections) instead of raw character 
    counts.

    Splits on markdown headers so sections stay intact. A section that's
    still bigger than chunk_size tokens falls back to the same sliding
    token-window approach as naive_fixed_size, so no chunk ever exceeds
    the limit.
    """

    if not doc.text.strip():
        return []

    if (overlap < 0) or (chunk_size <= overlap):
        raise ValueError("overlap must be a positive number and less than chunk_size")

    tokenizer_model = "cl100k_base"
    enc = tiktoken.get_encoding(tokenizer_model)

    sections = _split_by_headers(doc.text)

    chunks = []
    chunk_idx = 0

    for section in sections:
        section_text = section["text"]
        if not section_text.strip():
            continue

        heading_path = section["heading_path"]
        heading_prefix = " > ".join(heading_path)
        section_tokens = enc.encode(section_text)

        if len(section_tokens) <= chunk_size:
            sub_texts = [section_text]
        else:
            sub_texts = _split_section_preserving_table_headers(section_text, chunk_size, overlap, enc)

        for sub_text in sub_texts:
            chunk_text = f"{heading_prefix}\n\n{sub_text}" if heading_prefix else sub_text
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                text=chunk_text,
                metadata={
                    "source_path": doc.source_path,
                    "chunk_index": chunk_idx,
                    "token_count": len(enc.encode(chunk_text)),
                    "heading_path": heading_prefix,
                    **doc.metadata,
                },
            ))
            chunk_idx += 1

    return chunks

