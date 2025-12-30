"""
Parse Google Cloud Vision API responses into structured data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from google.cloud.vision_v1 import types


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    vertices: List[Tuple[int, int]]
    
    @property
    def x_min(self) -> int:
        return min(v[0] for v in self.vertices)
    
    @property
    def y_min(self) -> int:
        return min(v[1] for v in self.vertices)
    
    @property
    def x_max(self) -> int:
        return max(v[0] for v in self.vertices)
    
    @property
    def y_max(self) -> int:
        return max(v[1] for v in self.vertices)
    
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    def to_dict(self) -> dict:
        return {
            "vertices": self.vertices,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height
        }


@dataclass
class TextSymbol:
    """Individual character/symbol."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None
        }


@dataclass
class TextWord:
    """Word with symbols."""
    text: str
    confidence: float
    symbols: List[TextSymbol] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "symbols": [s.to_dict() for s in self.symbols],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None
        }


@dataclass
class TextLine:
    """Line of text with words."""
    text: str
    confidence: float
    words: List[TextWord] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None
        }


@dataclass
class TextBlock:
    """Block of text (paragraph)."""
    text: str
    confidence: float
    lines: List[TextLine] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    block_type: str = "TEXT"
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "block_type": self.block_type,
            "lines": [l.to_dict() for l in self.lines],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None
        }


@dataclass
class TextPage:
    """Page of text with blocks."""
    width: int
    height: int
    blocks: List[TextBlock] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "blocks": [b.to_dict() for b in self.blocks]
        }


@dataclass
class OCRResult:
    """Complete OCR result."""
    text: str
    confidence: float
    pages: List[TextPage] = field(default_factory=list)
    language: str = ""
    detection_type: str = ""
    error: Optional[str] = None
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @property
    def blocks(self) -> List[TextBlock]:
        """Flatten all blocks from all pages."""
        return [block for page in self.pages for block in page.blocks]
    
    @property
    def lines(self) -> List[TextLine]:
        """Flatten all lines from all blocks."""
        return [line for block in self.blocks for line in block.lines]
    
    @property
    def words(self) -> List[TextWord]:
        """Flatten all words from all lines."""
        return [word for line in self.lines for word in line.words]
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "detection_type": self.detection_type,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "pages": [p.to_dict() for p in self.pages],
            "error": self.error
        }
    
    def to_plain_text(self) -> str:
        """Return just the extracted text."""
        return self.text
    
    def to_hocr(self) -> str:
        """Convert to hOCR format."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
            '<head>',
            '<title>OCR Output</title>',
            '<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>',
            '<meta name="ocr-system" content="cloud-vision-ocr"/>',
            '</head>',
            '<body>',
        ]
        
        for page_idx, page in enumerate(self.pages):
            lines.append(f'<div class="ocr_page" id="page_{page_idx}" title="bbox 0 0 {page.width} {page.height}">')
            
            for block_idx, block in enumerate(page.blocks):
                if block.bounding_box:
                    bbox = f"bbox {block.bounding_box.x_min} {block.bounding_box.y_min} {block.bounding_box.x_max} {block.bounding_box.y_max}"
                else:
                    bbox = ""
                lines.append(f'<div class="ocr_carea" id="block_{page_idx}_{block_idx}" title="{bbox}">')
                
                for line_idx, line in enumerate(block.lines):
                    if line.bounding_box:
                        bbox = f"bbox {line.bounding_box.x_min} {line.bounding_box.y_min} {line.bounding_box.x_max} {line.bounding_box.y_max}"
                    else:
                        bbox = ""
                    lines.append(f'<span class="ocr_line" id="line_{page_idx}_{block_idx}_{line_idx}" title="{bbox}">')
                    
                    for word_idx, word in enumerate(line.words):
                        if word.bounding_box:
                            bbox = f"bbox {word.bounding_box.x_min} {word.bounding_box.y_min} {word.bounding_box.x_max} {word.bounding_box.y_max}"
                        else:
                            bbox = ""
                        conf = f"x_wconf {int(word.confidence * 100)}"
                        lines.append(f'<span class="ocrx_word" id="word_{page_idx}_{block_idx}_{line_idx}_{word_idx}" title="{bbox}; {conf}">{word.text}</span>')
                    
                    lines.append('</span>')
                lines.append('</div>')
            lines.append('</div>')
        
        lines.extend(['</body>', '</html>'])
        return '\n'.join(lines)


def _parse_bounding_poly(bounding_poly) -> Optional[BoundingBox]:
    """Parse Vision API bounding poly to BoundingBox."""
    if not bounding_poly or not bounding_poly.vertices:
        return None
    
    vertices = [(v.x, v.y) for v in bounding_poly.vertices]
    return BoundingBox(vertices=vertices)


def parse_vision_response(
    response: types.AnnotateImageResponse,
    detection_type: str
) -> OCRResult:
    """
    Parse Vision API response into OCRResult.
    
    Args:
        response: Vision API response
        detection_type: The detection type used
    
    Returns:
        OCRResult with structured text data
    """
    # Handle simple TEXT_DETECTION
    if detection_type == "TEXT_DETECTION":
        annotations = response.text_annotations
        if not annotations:
            return OCRResult(text="", confidence=0.0, detection_type=detection_type)
        
        # First annotation contains full text
        full_text = annotations[0].description if annotations else ""
        
        # Calculate average confidence (Vision API doesn't provide for TEXT_DETECTION)
        return OCRResult(
            text=full_text.strip(),
            confidence=1.0,  # TEXT_DETECTION doesn't provide confidence
            detection_type=detection_type
        )
    
    # Handle DOCUMENT_TEXT_DETECTION with full structure
    full_text_annotation = response.full_text_annotation
    
    if not full_text_annotation or not full_text_annotation.text:
        return OCRResult(text="", confidence=0.0, detection_type=detection_type)
    
    pages = []
    all_confidences = []
    detected_language = ""
    
    for page in full_text_annotation.pages:
        # Get detected language
        if page.property and page.property.detected_languages:
            lang = page.property.detected_languages[0]
            detected_language = lang.language_code
        
        page_blocks = []
        
        for block in page.blocks:
            block_lines = []
            block_text_parts = []
            
            for paragraph in block.paragraphs:
                para_words = []
                para_text_parts = []
                
                for word in paragraph.words:
                    word_symbols = []
                    word_text = ""
                    word_confidence = 0.0
                    
                    for symbol in word.symbols:
                        word_text += symbol.text
                        if symbol.confidence:
                            all_confidences.append(symbol.confidence)
                        
                        word_symbols.append(TextSymbol(
                            text=symbol.text,
                            confidence=symbol.confidence or 0.0,
                            bounding_box=_parse_bounding_poly(symbol.bounding_box)
                        ))
                    
                    word_confidence = word.confidence if word.confidence else (
                        sum(s.confidence for s in word_symbols) / len(word_symbols) if word_symbols else 0.0
                    )
                    
                    para_words.append(TextWord(
                        text=word_text,
                        confidence=word_confidence,
                        symbols=word_symbols,
                        bounding_box=_parse_bounding_poly(word.bounding_box)
                    ))
                    para_text_parts.append(word_text)
                
                # Create line from paragraph (simplification)
                para_text = " ".join(para_text_parts)
                para_confidence = paragraph.confidence if paragraph.confidence else (
                    sum(w.confidence for w in para_words) / len(para_words) if para_words else 0.0
                )
                
                block_lines.append(TextLine(
                    text=para_text,
                    confidence=para_confidence,
                    words=para_words,
                    bounding_box=_parse_bounding_poly(paragraph.bounding_box)
                ))
                block_text_parts.append(para_text)
            
            block_text = "\n".join(block_text_parts)
            block_confidence = block.confidence if block.confidence else (
                sum(l.confidence for l in block_lines) / len(block_lines) if block_lines else 0.0
            )
            
            # Map block type
            block_type_map = {
                0: "UNKNOWN",
                1: "TEXT",
                2: "TABLE",
                3: "PICTURE",
                4: "RULER",
                5: "BARCODE"
            }
            block_type = block_type_map.get(block.block_type, "TEXT")
            
            page_blocks.append(TextBlock(
                text=block_text,
                confidence=block_confidence,
                lines=block_lines,
                bounding_box=_parse_bounding_poly(block.bounding_box),
                block_type=block_type
            ))
        
        page_confidence = sum(b.confidence for b in page_blocks) / len(page_blocks) if page_blocks else 0.0
        
        pages.append(TextPage(
            width=page.width,
            height=page.height,
            blocks=page_blocks,
            confidence=page_confidence
        ))
    
    overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    return OCRResult(
        text=full_text_annotation.text.strip(),
        confidence=overall_confidence,
        pages=pages,
        language=detected_language,
        detection_type=detection_type
    )
