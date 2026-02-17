#!/usr/bin/env python3
"""
mac_ocr.py - CLI for document OCR using macOS Vision

Extracts text from document images (converted PDFs, photos, scans, etc.)
using macOS Vision library via PyObjC.

Requirements:
    pip install opencv-python numpy pillow pyobjc-framework-Vision pyobjc-framework-Quartz

Usage:
    mac-ocr documento.jpg
    mac-ocr documento.jpg --output texto.txt
    mac-ocr documento.jpg --format json --min-confidence 0.5
    mac-ocr *.jpg --output-dir resultados/
"""

import argparse
import json
import sys
import os
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

# macOS Vision OCR via PyObjC
from Vision import VNRecognizeTextRequest, VNImageRequestHandler


def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Loads an image from file and returns as OpenCV BGR array.
    
    Returns:
        (image_bgr, original_width, original_height)
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    return img, w, h


def ocr_vision(
    image_bgr: np.ndarray,
    min_conf: float = 0.0,
    languages: Optional[List[str]] = None,
    accurate: bool = True,
    scale: float = 1.0
) -> List[Dict]:
    """
    Performs OCR on an image using macOS Vision.
    
    Args:
        image_bgr: Image in BGR format (OpenCV)
        min_conf: Minimum confidence to include results (0.0-1.0)
        languages: List of language codes (e.g., ["pt-BR", "en-US"])
        accurate: If True, uses accurate mode (slower), otherwise fast
        scale: Scale factor to improve OCR (1.0 = no scaling)
    
    Returns:
        List of dictionaries with text, confidence and normalized bounding box
    """
    # Apply scale if necessary
    img = image_bgr
    if scale and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to RGB and then to PIL Image
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    
    # Convert to PNG bytes
    import io
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    img_data = buf.getvalue()
    
    results = []
    
    def handler(request, error):
        if error is not None:
            return
        obs = request.results() or []
        for o in obs:
            cands = o.topCandidates_(1)
            if not cands:
                continue
            cand = cands[0]
            text = str(cand.string())
            conf = float(cand.confidence())
            if conf < min_conf:
                continue
            
            # Normalize accents (avoids strange combined characters)
            text = unicodedata.normalize("NFC", text)
            
            bb = o.boundingBox()  # normalized, bottom-left origin
            results.append({
                "text": text,
                "confidence": conf,
                "bbox_norm": {
                    "x": float(bb.origin.x),
                    "y": float(bb.origin.y),
                    "w": float(bb.size.width),
                    "h": float(bb.size.height)
                },
            })
    
    req = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    
    # 0=fast, 1=accurate
    req.setRecognitionLevel_(1 if accurate else 0)
    req.setUsesLanguageCorrection_(True)
    
    # Configure languages if provided
    if languages:
        req.setRecognitionLanguages_(languages)
    
    h = VNImageRequestHandler.alloc().initWithData_options_(img_data, None)
    h.performRequests_error_([req], None)
    
    return results


def norm_to_pixels(bbox_norm: Dict, img_w: int, img_h: int) -> Dict:
    """
    Converts normalized bounding box (bottom-left origin) to pixels (top-left origin).
    """
    x = bbox_norm["x"] * img_w
    y_bl = bbox_norm["y"] * img_h
    w = bbox_norm["w"] * img_w
    h = bbox_norm["h"] * img_h
    y_tl = img_h - (y_bl + h)
    return {
        "x": int(round(x)),
        "y": int(round(y_tl)),
        "w": int(round(w)),
        "h": int(round(h))
    }


def sort_by_position(items: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    """
    Sorts OCR items by position: first by Y (line), then by X (column).
    Useful for preserving reading order in documents.
    """
    def get_sort_key(item):
        px = norm_to_pixels(item["bbox_norm"], img_w, img_h)
        # Group by line (tolerance of ~5% of image height)
        line_tolerance = img_h * 0.05
        line = int(px["y"] / line_tolerance)
        return (line, px["x"])
    
    return sorted(items, key=get_sort_key)


def format_as_text(items: List[Dict], preserve_layout: bool = True, img_w: int = 0, img_h: int = 0) -> str:
    """
    Formats OCR results as plain text.
    
    Args:
        items: List of sorted OCR items (with bbox_norm)
        preserve_layout: If True, tries to preserve line breaks and spacing
        img_w: Image width (to calculate positions)
        img_h: Image height (to calculate positions)
    """
    if not items:
        return ""
    
    if not preserve_layout or not img_w or not img_h:
        # Just concatenate everything
        return " ".join(item["text"] for item in items)
    
    # Try to preserve layout based on Y positions
    lines = []
    current_line = []
    last_y = None
    line_tolerance = img_h * 0.03  # 3% of image height
    
    for item in items:
        px = norm_to_pixels(item["bbox_norm"], img_w, img_h)
        text = item["text"]
        y = px["y"]
        
        # If significantly changed line, finalize previous line
        if last_y is not None and abs(y - last_y) > line_tolerance:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
        
        current_line.append(text)
        last_y = y
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def process_image(
    image_path: str,
    min_conf: float = 0.0,
    languages: Optional[List[str]] = None,
    accurate: bool = True,
    scale: float = 1.0,
    preserve_layout: bool = True
) -> Dict:
    """
    Processes a single image and returns OCR results.
    """
    try:
        image_bgr, orig_w, orig_h = load_image(image_path)
        
        # Perform OCR
        ocr_items = ocr_vision(
            image_bgr,
            min_conf=min_conf,
            languages=languages,
            accurate=accurate,
            scale=scale
        )
        
        # Calculate dimensions after scale
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        
        # Convert bounding boxes to pixels and sort
        items_with_bbox = []
        for item in ocr_items:
            px = norm_to_pixels(item["bbox_norm"], scaled_w, scaled_h)
            # Adjust to original size (divide by scale)
            px_orig = {k: int(round(v / scale)) for k, v in px.items()}
            items_with_bbox.append({
                "text": item["text"],
                "confidence": item["confidence"],
                "bbox": px_orig
            })
        
        # Sort by position (line, then column)
        sorted_items = sort_by_position(ocr_items, scaled_w, scaled_h)
        
        # Extract plain text (preserving layout if requested)
        text_content = format_as_text(
            sorted_items,
            preserve_layout=preserve_layout,
            img_w=scaled_w,
            img_h=scaled_h
        )
        
        return {
            "ok": True,
            "file": image_path,
            "image_size": {"width": orig_w, "height": orig_h},
            "items_count": len(items_with_bbox),
            "text": text_content,
            "items": items_with_bbox
        }
    
    except Exception as e:
        return {
            "ok": False,
            "file": image_path,
            "error": str(e)
        }


def main():
    ap = argparse.ArgumentParser(
        description="Document OCR using macOS Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text from an image
  %(prog)s documento.jpg
  
  # Save to text file
  %(prog)s documento.jpg --output resultado.txt
  
  # JSON output with details
  %(prog)s documento.jpg --format json
  
  # Multiple files
  %(prog)s *.jpg --output-dir resultados/
  
  # Adjust minimum confidence and languages
  %(prog)s documento.jpg --min-confidence 0.5 --languages pt-BR,en-US
  
  # Fast mode (less accurate)
  %(prog)s documento.jpg --level fast
        """
    )
    
    ap.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image(s) to process"
    )
    
    ap.add_argument(
        "--output", "-o",
        help="Output file (text only). If not specified, prints to stdout"
    )
    
    ap.add_argument(
        "--output-dir",
        help="Directory to save results (creates .txt file for each image)"
    )
    
    ap.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json"
    )
    
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence to include results (0.0-1.0, default: 0.0)"
    )
    
    ap.add_argument(
        "--languages",
        help="Languages separated by comma (e.g., pt-BR,pt-PT,en-US). Default: pt-BR,pt-PT,en-US"
    )
    
    ap.add_argument(
        "--level",
        choices=["fast", "accurate"],
        default="accurate",
        help="Accuracy level: fast (quick) or accurate (precise, default)"
    )
    
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to improve OCR (e.g., 2.0 to double size). Default: 1.0"
    )
    
    ap.add_argument(
        "--no-preserve-layout",
        action="store_true",
        help="Do not try to preserve layout/line breaks (just concatenate text)"
    )
    
    args = ap.parse_args()
    
    # Process languages
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]
    else:
        languages = ["pt-BR", "pt-PT", "en-US"]
    
    # Process each image
    results = []
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}", file=sys.stderr)
            results.append({
                "ok": False,
                "file": image_path,
                "error": "File not found"
            })
            continue
        
        result = process_image(
            image_path,
            min_conf=args.min_confidence,
            languages=languages,
            accurate=(args.level == "accurate"),
            scale=args.scale,
            preserve_layout=not args.no_preserve_layout
        )
        results.append(result)
    
    # Format and save output
    if args.output_dir:
        # Save each result in a separate file
        os.makedirs(args.output_dir, exist_ok=True)
        for result in results:
            if not result["ok"]:
                continue
            
            base_name = Path(result["file"]).stem
            output_file = os.path.join(args.output_dir, f"{base_name}.txt")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"Saved: {output_file}", file=sys.stderr)
    
    elif args.output:
        # Save everything to a single file
        if args.format == "json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            # Concatenate text from all results
            texts = []
            for result in results:
                if result["ok"]:
                    texts.append(f"=== {result['file']} ===\n\n{result['text']}\n")
                else:
                    texts.append(f"=== {result['file']} ===\n\nError: {result.get('error', 'Unknown')}\n")
            
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
    
    else:
        # Print to stdout
        if args.format == "json":
            json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        else:
            # Print text from each result
            for result in results:
                if result["ok"]:
                    if len(results) > 1:
                        print(f"=== {result['file']} ===\n", file=sys.stdout)
                    print(result["text"], file=sys.stdout)
                    if len(results) > 1:
                        print("\n", file=sys.stdout)
                else:
                    print(f"Error processing {result['file']}: {result.get('error', 'Unknown')}", file=sys.stderr)


if __name__ == "__main__":
    main()
