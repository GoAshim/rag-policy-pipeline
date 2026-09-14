"""
Ingestion: turn raw files in data/raw/ into clean, plain-text documents,
each tagged with metadata about where it came from.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

# For PDF extraction
import pymupdf
import pymupdf4llm

# For HTML extraction
from bs4 import BeautifulSoup, FeatureNotFound
import html2text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
REGISTRY_PATH = DATA_DIR / "processed" / "doc_registry.json"

@dataclass
class RawDocument:
    """A single ingested document, before chunking."""
    source_path: str    # where this came from (file path or URL)
    doc_id: str         # a stable identifier you assign
    text: str           # cleaned plain text
    metadata: dict      # e.g. {"source": "cms.gov", "title": ..., "format": "pdf"}


def load_pdf(path: Path) -> tuple[bool, str, dict, str]:
    """
    Extracts and returns all text from a PDF file using PyMuPDF4llm.

    Args:
        file_name (str): Name to the PDF file.

    Returns:
        success: boolean indicating if the content was extracted
        error_message: string with error message during extracting text from the PDF.
        metadata: dictionary with different attributes of the file.
        content: string with extracted text from the PDF.

    Error Handling:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
        RuntimeError: If text extraction fails.
    """

    success = False
    error_message = ""
    metadata = {}
    content = ""
    text_page_count = 0
    scanned_page_count = 0

    # Validate file extension
    if not path.name.lower().endswith(".pdf"):
        error_message = f"The file {path.name} must be a PDF."
        return success, error_message, metadata, content

    # Validate file existence
    if not path.is_file():
        error_message = f"File not found: {path}"
        return success, error_message, metadata, content

    try:
        with pymupdf.open(path) as pdf:
            title = pdf.metadata.get("title") or path.name
            total_page_count = len(pdf)
            text_pages = []

            for page_num in range(total_page_count):
                page = pdf[page_num]

                # Check if the page have extractable text
                if page.get_text().strip():
                    text_page_count += 1
                    text_pages.append(page_num)
                else:
                    scanned_page_count += 1

            # Pass the open doc and the pages with extractable text as Markdown
            if text_pages:
                file_md = pymupdf4llm.to_markdown(pdf, pages=text_pages, use_ocr=False)
            else:
                file_md = ""

            success = True
            content = file_md.strip()
            metadata = {
                "title": title,
                "total_page_count": total_page_count,
                "text_page_count": text_page_count,
                "scanned_page_count": scanned_page_count,
            }

            return success, error_message, metadata, content

    except Exception as e:
        error_message = f"[ERROR] Could not open or read '{path.name}': {e}"
        return success, error_message, metadata, content


def load_html(path: Path) -> tuple[bool, str, dict, str]:
    """
    Extract the main readable text from a saved HTML file, stripping
    nav bars, footers, and other boilerplate.

    Args:
        file_name (Path): Path to the HTML file.

    Returns:
        success: boolean indicating if the content was extracted
        error_message: string with error message during extracting text from the HTML.
        metadata: dictionary with different attributes of the file.
        content: string with extracted text from the HTML.

    Error Handling:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a HTML.
        RuntimeError: If text extraction fails.
    """

    success = False
    error_message = ""
    metadata = {}
    content = ""

    # Validate file extension
    if not path.name.lower().endswith(".html"):
        error_message = f"The file {path.name} must be a HTML."
        return success, error_message, metadata, content

    # Validate file existence
    if not path.is_file():
        error_message = f"File not found: {path}"
        return success, error_message, metadata, content

    # Validate if the file has any content
    if path.stat().st_size == 0:
        error_message = f"Skipping empty HTML file: {path}"
        return success, error_message, metadata, content

    # Read the content of the HTML file using BeautifulSoup
    try:
        with open(path, "rb") as f:
            raw_html = f.read()
    except (PermissionError, OSError) as e:
        error_message = f"I/O error reading {path}: {e}"
        return success, error_message, metadata, content

    # HTML Parsing (with parser fallback)
    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except FeatureNotFound:
        # lxml not found, falling back to html.parser
        soup = BeautifulSoup(raw_html, "html.parser")
    except Exception as e:
        error_message = f"Failed to parse HTML for {path}: {e}"
        return success, error_message, metadata, content

    # Data Extraction & Cleanup
    try:
        # Extract title safely
        if soup.title:
            title = soup.title.get_text(strip=True)
        else:
            title = path.stem

        # Extract meta description safely
        meta_desc = ""
        desc_tag = (
            soup.find("meta", attrs={"name": "description"}) 
            or soup.find("meta", attrs={"property": "og:description"})
        )
        if desc_tag and desc_tag.get("content"):
            meta_desc = str(desc_tag["content"]).strip()

        # Remove boilerplate tags
        unwanted_tags = [
            "script", "style", "noscript", "nav", "footer", 
            "header", "aside", "svg", "form", "iframe"
        ]
        for tag in soup(unwanted_tags):
            tag.decompose()

        # Locate primary content and convert to Markdown
        h2t = html2text.HTML2Text()
        h2t.ignore_links = True
        h2t.ignore_images = True
        h2t.body_width = 0
        h2t.protect_links = True
        main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
        markdown_text = h2t.handle(str(main_content)).strip()

        # Validate extraction output
        if not markdown_text:
            error_message = f"No usable content extracted from {path}"
            return success, error_message, metadata, content

        # Gather the return elements of the method
        success = True
        content = markdown_text
        metadata = {
            "source": str(path.resolve()),
            "title": title,
            "description": meta_desc,
        }

        return success, error_message, metadata, content

    except Exception as e:
        error_message = f"Unexpected error processing DOM for {path}: {e}"
        return success, error_message, metadata, content


def generate_content_doc_id(path: Path, prefix: str = "doc_") -> str:
    """
    read the raw bytes into memory and hash them directly, without any parsing.
    For larger files, we should read in fixed-size blocks and feed them to the 
    hash incrementally instead of loading the whole file at once.
    """
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{prefix}{content_hash[:16]}"


def load_doc_registry(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_doc_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def load_corpus(raw_dir: Path) -> list[RawDocument]:
    """
    Walk raw_dir, dispatch each file to the right loader based on
    extension, and return a list of RawDocument objects. Make sure
    to skip a file that's already been loaded.
    """

    documents = []
    failures = []
    registry = load_doc_registry(REGISTRY_PATH)

    for file_path in raw_dir.iterdir():

        doc_id=generate_content_doc_id(file_path)

        if doc_id in registry:
            cached = registry[doc_id]
            documents.append(RawDocument(
                source_path=cached["source_path"],
                doc_id=doc_id,
                text=cached["text"],
                metadata=cached["metadata"],
            ))
            continue
        else:
            # Ingest the file
            if file_path.suffix.lower() == ".pdf":
                success, error_message, metadata, content = load_pdf(file_path)
                fmt = "pdf"
            elif file_path.suffix.lower() == ".html":
                success, error_message, metadata, content = load_html(file_path)
                fmt = "html"
            else:
                failures.append((file_path.name, "File extension is neither pdf or html"))
                continue  

            if not success:
                failures.append((file_path.name, error_message))
                continue

            metadata["format"] = fmt
            documents.append(RawDocument(
                source_path=str(file_path),
                doc_id=doc_id,
                text=content,
                metadata=metadata,
            ))

            # Add the new doc id to the list of doc ids
            registry[doc_id] = {
                "source_path": str(file_path), 
                "format": fmt,
                "text":content,
                "metadata":metadata
            }

    save_doc_registry(REGISTRY_PATH, registry)
    

    if failures:
        print(f"\n{len(failures)} file(s) failed to ingest:")
        for filename, error in failures:
            print(f"\n {filename}: {error}")

    return documents


if __name__ == "__main__":
    docs = load_corpus(RAW_DIR)
    print(f"Loaded {len(docs)} documents")

