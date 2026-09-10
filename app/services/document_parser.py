import csv
import io
import re
from pathlib import Path
from pypdf import PdfReader
import docx
import olefile

class DocumentParser:
    @classmethod
    def parse_file(cls, file_path):
        """
        Parses a document file (PDF, DOC, DOCX, CSV) and returns a list of text sections
        with metadata (e.g., page/section, source name).
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return cls._parse_pdf(path)
        elif ext == ".docx":
            return cls._parse_docx(path)
        elif ext == ".doc":
            return cls._parse_doc(path)
        elif ext == ".csv":
            return cls._parse_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, .doc, .csv")

    @classmethod
    def _parse_pdf(cls, path):
        sections = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                cleaned = cls._clean_text(page_text)
                if cleaned:
                    sections.append({
                        "content": cleaned,
                        "metadata": {
                            "source_file": path.name,
                            "type": "pdf",
                            "page": idx + 1,
                            "total_pages": total_pages,
                            "section_title": f"Page {idx + 1}"
                        }
                    })
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {path.name}: {str(e)}")
        return sections

    @classmethod
    def _parse_docx(cls, path):
        sections = []
        try:
            doc = docx.Document(str(path))
            
            # Extract paragraphs
            para_texts = []
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    para_texts.append(txt)

            # Extract tables
            table_texts = []
            for t_idx, table in enumerate(doc.tables):
                t_lines = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    # Deduplicate adjacent identical cells if merged
                    dedup_cells = []
                    for c in row_cells:
                        if not dedup_cells or dedup_cells[-1] != c:
                            dedup_cells.append(c)
                    if any(dedup_cells):
                        t_lines.append(" | ".join(dedup_cells))
                if t_lines:
                    table_texts.append(f"[Table {t_idx + 1}]\n" + "\n".join(t_lines))

            full_text = "\n\n".join(para_texts + table_texts)
            cleaned = cls._clean_text(full_text)
            if cleaned:
                sections.append({
                    "content": cleaned,
                    "metadata": {
                        "source_file": path.name,
                        "type": "docx",
                        "section_title": "Document Content"
                    }
                })
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX {path.name}: {str(e)}")
        return sections

    @classmethod
    def _parse_doc(cls, path):
        """
        Extracts text from legacy .doc (OLE2 Compound File) binary format.
        Uses olefile stream inspection and printable string heuristics.
        """
        sections = []
        try:
            if olefile.isOleFile(str(path)):
                with olefile.OleFileIO(str(path)) as ole:
                    if ole.exists("WordDocument"):
                        stream = ole.openstream("WordDocument").read()
                        # Extract UTF-16 / ASCII printable sequences
                        # Typically Word stores text in 1-byte or 2-byte chunks
                        text = stream.decode("latin1", errors="ignore")
                        # Filter readable words
                        cleaned_words = re.findall(r'[a-zA-Z0-9\s.,!?;:\-–—\(\)\[\]"\'@#$%&*+/=<>_]{4,}', text)
                        clean_content = " ".join(cleaned_words)
                        clean_content = cls._clean_text(clean_content)
                        if clean_content:
                            sections.append({
                                "content": clean_content,
                                "metadata": {
                                    "source_file": path.name,
                                    "type": "doc",
                                    "section_title": "Legacy Word Document"
                                }
                            })
                            return sections
            
            # Fallback for plain text or pseudo-doc
            with open(path, "rb") as f:
                raw = f.read().decode("latin1", errors="ignore")
                words = re.findall(r'[a-zA-Z0-9\s.,!?;:\-–—\(\)\[\]"\'@#$%&*+/=<>_]{4,}', raw)
                clean_content = cls._clean_text(" ".join(words))
                if clean_content:
                    sections.append({
                        "content": clean_content,
                        "metadata": {
                            "source_file": path.name,
                            "type": "doc",
                            "section_title": "Binary Document Stream"
                        }
                    })
        except Exception as e:
            raise RuntimeError(f"Failed to parse legacy DOC {path.name}: {str(e)}")
        return sections

    @classmethod
    def _parse_csv(cls, path):
        sections = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                headers = []
                try:
                    headers = [h.strip() for h in next(reader)]
                except StopIteration:
                    return []

                rows_batch = []
                for row_idx, row in enumerate(reader, start=1):
                    row_parts = []
                    for h, val in zip(headers, row):
                        val_str = val.strip()
                        if val_str:
                            row_parts.append(f"{h}: {val_str}")
                    if row_parts:
                        rows_batch.append(f"Row {row_idx}: " + "; ".join(row_parts))

                    # Batch rows into manageable blocks for RAG
                    if len(rows_batch) >= 20:
                        content = "\n".join(rows_batch)
                        sections.append({
                            "content": content,
                            "metadata": {
                                "source_file": path.name,
                                "type": "csv",
                                "headers": ", ".join(headers),
                                "section_title": f"Rows {row_idx - len(rows_batch) + 1} - {row_idx}"
                            }
                        })
                        rows_batch = []

                if rows_batch:
                    content = "\n".join(rows_batch)
                    sections.append({
                        "content": content,
                        "metadata": {
                            "source_file": path.name,
                            "type": "csv",
                            "headers": ", ".join(headers),
                            "section_title": "CSV Data Records"
                        }
                    })
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV {path.name}: {str(e)}")
        return sections

    @staticmethod
    def _clean_text(text):
        if not text:
            return ""
        # Remove repeated whitespace and newlines
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
