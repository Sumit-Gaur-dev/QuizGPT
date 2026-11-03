"""
PDF Ingestion and Text Extraction Module.
Extracts text and page-level metadata from PDF documents using PyMuPDF.
Strictly decoupled from Streamlit UI logic.
"""

from typing import Any
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from config.settings import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS


def validate_pdf(filename: str, file_size_bytes: int) -> tuple[bool, str | None]:
    """
    Validate PDF filename extension and file size.
    
    Args:
        filename: Name of the uploaded file.
        file_size_bytes: Size of the file in bytes.
        
    Returns:
        tuple of (is_valid, error_message).
    """
    # Check file extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, f"Invalid file format. Only PDF files ({', '.join(ALLOWED_EXTENSIONS)}) are supported."

    # Check empty file
    if file_size_bytes <= 0:
        return False, "File is empty (0 bytes). Please upload a valid PDF document."

    # Check maximum allowed size
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File size ({round(file_size_bytes / (1024 * 1024), 2)} MB) exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."

    return True, None


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Extract text page-by-page from raw PDF bytes using PyMuPDF.
    
    Args:
        file_bytes: Raw binary content of the PDF file.
        filename: Original name of the PDF file.
        
    Returns:
        dict containing extraction status, metadata, and extracted page records:
        {
            "success": bool,
            "filename": str,
            "total_pages": int,
            "extracted_pages_count": int,
            "pages": [
                {
                    "page_number": int,
                    "text": str,
                    "char_count": int
                }, ...
            ],
            "total_characters": int,
            "warning": str | None,
            "error": str | None
        }
    """
    # 1. Preliminary validation
    is_valid, validation_error = validate_pdf(filename, len(file_bytes))
    if not is_valid:
        return {
            "success": False,
            "filename": filename,
            "total_pages": 0,
            "extracted_pages_count": 0,
            "pages": [],
            "total_characters": 0,
            "warning": None,
            "error": validation_error
        }

    # 2. Open PDF document stream
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        return {
            "success": False,
            "filename": filename,
            "total_pages": 0,
            "extracted_pages_count": 0,
            "pages": [],
            "total_characters": 0,
            "warning": None,
            "error": f"Failed to open PDF document. File may be corrupted or encrypted: {str(exc)}"
        }

    try:
        total_pages = doc.page_count
        if total_pages == 0:
            return {
                "success": False,
                "filename": filename,
                "total_pages": 0,
                "extracted_pages_count": 0,
                "pages": [],
                "total_characters": 0,
                "warning": None,
                "error": "The PDF document contains 0 pages."
            }

        extracted_pages = []

        # 3. Extract text page-by-page
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()

            # Only retain non-empty text pages
            if page_text:
                extracted_pages.append({
                    "page_number": page_idx + 1,  # 1-indexed for human readability
                    "text": page_text,
                    "char_count": len(page_text)
                })

        # 4. Check for scanned / image-only PDF
        if len(extracted_pages) == 0:
            return {
                "success": False,
                "filename": filename,
                "total_pages": total_pages,
                "extracted_pages_count": 0,
                "pages": [],
                "total_characters": 0,
                "warning": None,
                "error": (
                    "No extractable text found in this document. It appears to be a scanned or image-only PDF. "
                    "Optical Character Recognition (OCR) is not supported in V1."
                )
            }

        # 5. Handle partial extraction (some pages without text)
        warning_msg = None
        if len(extracted_pages) < total_pages:
            skipped_pages = total_pages - len(extracted_pages)
            warning_msg = f"Extracted text from {len(extracted_pages)} of {total_pages} page(s). {skipped_pages} page(s) were blank or image-only and skipped."

        total_chars = sum(p["char_count"] for p in extracted_pages)

        return {
            "success": True,
            "filename": filename,
            "total_pages": total_pages,
            "extracted_pages_count": len(extracted_pages),
            "pages": extracted_pages,
            "total_characters": total_chars,
            "warning": warning_msg,
            "error": None
        }

    finally:
        doc.close()

