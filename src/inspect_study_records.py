"""
Comprehensive parser for First EF5 RT Study spreadsheets (.xlsx), Word reports (.docx),
PowerPoint decks (.pptx), 15 Dec mouse_set_2a.roi, and histogram cross-matching.
Uses Python stdlib (zipfile + xml) so no extra packages beyond the base image are needed.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.rt_image_loader import build_roi_3d_mask, load_rt_image_dicom, load_rt_image_roi


# ---------------------------------------------------------------------------
# Minimal stdlib XLSX reader (handles sharedStrings, inlineStr, numbers, dates)
# ---------------------------------------------------------------------------
NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _col_to_idx(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    if not m:
        return 0
    idx = 0
    for ch in m.group(1):
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_sheets(xlsx_path: Path) -> Dict[str, List[List[str]]]:
    out: Dict[str, List[List[str]]] = {}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        # Shared strings
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS_MAIN}si"):
                txt = "".join(t.text or "" for t in si.iter(f"{NS_MAIN}t"))
                shared.append(txt)

        # Workbook sheet names + rIds
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{NS_PKG_REL}Relationship")
        }

        for sheet in wb.findall(f".//{NS_MAIN}sheet"):
            name = sheet.attrib["name"]
            rid = sheet.attrib.get(f"{NS_REL}id", "")
            target = rid_to_target.get(rid, "")
            if not target:
                continue
            target_path = target.lstrip("/")
            if not target_path.startswith("xl/"):
                target_path = "xl/" + target_path
            if target_path not in zf.namelist():
                continue

            sroot = ET.fromstring(zf.read(target_path))
            rows_data: List[List[str]] = []
            for row in sroot.findall(f".//{NS_MAIN}sheetData/{NS_MAIN}row"):
                cells: Dict[int, str] = {}
                max_c = -1
                for c in row.findall(f"{NS_MAIN}c"):
                    ref = c.attrib.get("r", "")
                    ci = _col_to_idx(ref) if ref else (max_c + 1)
                    ctype = c.attrib.get("t", "")
                    val = ""
                    if ctype == "s":
                        v = c.find(f"{NS_MAIN}v")
                        if v is not None and v.text is not None:
                            val = shared[int(v.text)]
                    elif ctype == "inlineStr":
                        val = "".join(t.text or "" for t in c.iter(f"{NS_MAIN}t"))
                    else:
                        v = c.find(f"{NS_MAIN}v")
                        if v is not None and v.text is not None:
                            val = v.text
                    cells[ci] = val
                    if ci > max_c:
                        max_c = ci
                if max_c >= 0:
                    row_list = [cells.get(i, "") for i in range(max_c + 1)]
                    if any(x.strip() for x in row_list):
                        rows_data.append(row_list)
            out[name] = rows_data
    return out


def read_docx_text(docx_path: Path) -> Tuple[List[str], List[str]]:
    paras: List[str] = []
    comments: List[str] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        if "word/document.xml" in zf.namelist():
            root = ET.fromstring(zf.read("word/document.xml"))
            w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            for p in root.iter(f"{w_ns}p"):
                txt = "".join(t.text or "" for t in p.iter(f"{w_ns}t")).strip()
                if txt:
                    paras.append(txt)
        if "word/comments.xml" in zf.namelist():
            root = ET.fromstring(zf.read("word/comments.xml"))
            w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            for c in root.findall(f"{w_ns}comment"):
                author = c.attrib.get(f"{w_ns}author", "")
                txt = "".join(t.text or "" for t in c.iter(f"{w_ns}t")).strip()
                if txt:
                    comments.append(f"[{author}] {txt}")
    return paras, comments


def read_pptx_text(pptx_path: Path) -> List[Tuple[int, List[str]]]:
    slides: List[Tuple[int, List[str]]] = []
    with zipfile.ZipFile(pptx_path, "r") as zf:
        slide_names = [
            n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)
        ]
        slide_names.sort(key=lambda s: int(re.search(r"slide(\d+)\.xml$", s).group(1)))
        a_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        for sn in slide_names:
            num = int(re.search(r"slide(\d+)\.xml$", sn).group(1))
            root = ET.fromstring(zf.read(sn))
            lines = []
            for p in root.iter(f"{a_ns}p"):
                txt = "".join(t.text or "" for t in p.iter(f"{a_ns}t")).strip()
                if txt:
                    lines.append(txt)
            slides.append((num, lines))
    return slides


def inspect_15dec_rois(dec15_dir: Path) -> None:
    print("=" * 90)
    print("A. 15 DEC SET 2 ROIs & VOLUMES")
    print("=" * 90)
    pet_vol = load_rt_image_dicom(dec15_dir / "mouse_set_20001.dcm")
    ct_vol = load_rt_image_dicom(dec15_dir / "mouse_set_20002.dcm")
    print(f"PET: shape={pet_vol.shape}, spacing={pet_vol.spacing}, origin={pet_vol.origin}, date={pet_vol.study_date}, units={pet_vol.units}, range=[{pet_vol.data.min():.3f}, {pet_vol.data.max():.3f}]")
    print(f"CT : shape={ct_vol.shape}, spacing={ct_vol.spacing}, origin={ct_vol.origin}, date={ct_vol.study_date}, units={ct_vol.units}, range=[{ct_vol.data.min():.1f}, {ct_vol.data.max():.1f}]")
    print(f"CT orientation: {np.round(ct_vol.orientation, 4).tolist()}")

    x_mid = 0.5 * (float(pet_vol.x_coords.min()) + float(pet_vol.x_coords.max()))
    z_mid = 0.5 * (float(pet_vol.z_coords.min()) + float(pet_vol.z_coords.max()))

    for rname in ["mouse_set_2.roi", "mouse_set_2a.roi"]:
        rpath = dec15_dir / rname
        ver, rois = load_rt_image_roi(rpath)
        print(f"\n--- {rname} ({len(rois)} ROIs, {ver}) ---")
        for roi in rois:
            cent = roi.centroid_mm
            bmin, bmax = roi.bounds_mm
            horiz = "Left" if cent[0] >= x_mid else "Right"
            vert = "Top" if cent[2] >= z_mid else "Bottom"
            ctypes = ",".join(sorted({c.roi_type for c in roi.contours}))
            mask = build_roi_3d_mask(pet_vol, roi)
            vals = np.asarray(pet_vol.data[mask], dtype=np.float64)
            dz, dy, dx = pet_vol.spacing
            vol_cc = vals.size * (dx * dy * dz) / 1000.0
            mean_v = float(vals.mean()) if vals.size else float("nan")
            max_v = float(vals.max()) if vals.size else float("nan")
            print(
                f"  {roi.name:20s} | {ctypes:9s} N={len(roi.contours):2d} | {vert:6s}-{horiz:5s} "
                f"| cent=({cent[0]:6.1f},{cent[1]:6.1f},{cent[2]:6.1f}) | pts={vals.size:6d} vol={vol_cc:7.4f}cc mean={mean_v:.4f} max={max_v:.4f}"
            )


def cross_match_histograms(root: Path) -> None:
    print("\n" + "=" * 90)
    print("B. HISTOGRAM BYTE-FOR-BYTE CROSS-MATCH (m1..m6 <-> ear-punch IDs)")
    print("=" * 90)
    rep_post = root / "images/set3/reports/A549 RT histograms/003. 1 month post-rt"
    jan3_dir = root / "images/set3/Post RT/2. 03 Jan"

    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    jan_hashes: Dict[str, List[str]] = {}
    for p in sorted(jan3_dir.rglob("hist_*.txt")):
        h = md5(p)
        jan_hashes.setdefault(h, []).append(p.relative_to(jan3_dir).as_posix())

    for p in sorted(rep_post.glob("hist_*.txt")):
        h = md5(p)
        matches = jan_hashes.get(h, ["NO EXACT MATCH"])
        print(f"  {p.name:20s}  ==>  {matches}")


def dump_spreadsheets(rep_dir: Path) -> None:
    print("\n" + "=" * 90)
    print("C. SPREADSHEET CONTENTS")
    print("=" * 90)
    targets = [
        "08 EF5 RT PET and Knottin data.xlsx",
        "Pre and Post RT EF5 Analysis.xlsx",
        "08 Pre and Post RT EF5 Uptake.xlsx",
        "08 EF5 IR Study Growth Curves i.xlsx",
        "Pre and Post RT Histogram Analysis.xlsx",
    ]
    for fname in targets:
        p = rep_dir / fname
        if not p.exists():
            continue
        sheets = read_xlsx_sheets(p)
        print(f"\n### {fname} (sheets: {list(sheets.keys())})")
        for sname, rows in sheets.items():
            # Skip purely knottin / aluc sheets unless small
            if any(k in sname.lower() for k in ["knottin", "2.5f", "pearl"]):
                print(f"  [Skipped sheet: {sname} ({len(rows)} rows)]")
                continue
            print(f"\n  --- Sheet: {sname} ({len(rows)} non-empty rows) ---")
            for idx, r in enumerate(rows[:65]):
                # Trim trailing empty cells
                while r and not r[-1].strip():
                    r.pop()
                print(f"    R{idx+1:02d}: " + " | ".join(r[:18]))
            if len(rows) > 65:
                print(f"    ... ({len(rows)-65} more rows)")


def dump_docs_and_ppts(rep_dir: Path) -> None:
    print("\n" + "=" * 90)
    print("D. WORD DOCS & POWERPOINT TEXT")
    print("=" * 90)
    for dname in ["08a Project Report.docx", "08a Ted's Comments on Project Report.docx"]:
        p = rep_dir / dname
        if not p.exists():
            continue
        paras, comments = read_docx_text(p)
        print(f"\n### {dname} ({len(paras)} paras, {len(comments)} comments)")
        if dname == "08a Project Report.docx":
            for para in paras:
                print(f"  {para}")
        if comments:
            print("  --- COMMENTS ---")
            for c in comments:
                print(f"  {c}")

    for pname in ["Pre and Post RT EF5 Scans.pptx", "08a Project Report Figures.pptx"]:
        p = rep_dir / pname
        if not p.exists():
            continue
        slides = read_pptx_text(p)
        print(f"\n### {pname} ({len(slides)} slides)")
        for num, lines in slides:
            print(f"  Slide {num:02d}: " + " || ".join(lines))


def main() -> None:
    root = Path("/workspace")
    inspect_15dec_rois(root / "images/set3/Post RT/1. 15 Dec")
    cross_match_histograms(root)
    dump_spreadsheets(root / "images/set3/reports")
    dump_docs_and_ppts(root / "images/set3/reports")


if __name__ == "__main__":
    main()
