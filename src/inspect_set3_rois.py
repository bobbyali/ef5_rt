"""
Inspects all RT_Image .roi/.ROI files, DICOM volumes (.dcm), and raw microPET .img.hdr
headers inside images/set3/ to inventory available mice, their 3D spatial / 2x2 bed positions,
and associated tumour / muscle / body ROIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.rt_image_loader import (
    RTRoi,
    RTImageVolume,
    build_roi_3d_mask,
    load_rt_image_dicom,
    load_rt_image_roi,
    parse_inveon_hdr,
)


def summarize_hdr_files(set3_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for hdr_path in sorted(set3_dir.rglob("*.hdr")):
        hdr = parse_inveon_hdr(hdr_path)
        rel = hdr_path.relative_to(set3_dir).as_posix()
        rows.append(
            {
                "rel_path": rel,
                "modality": hdr.get("modality", ""),
                "scan_time": hdr.get("scan_time", ""),
                "file_name": hdr.get("file_name", ""),
                "study": hdr.get("study", ""),
                "subject": hdr.get("subject_identifier", hdr.get("subject", "")),
                "dims": f"{hdr.get('z_dimension','?')}x{hdr.get('y_dimension','?')}x{hdr.get('x_dimension','?')}",
                "voxel_mm": f"{hdr.get('pixel_size_z','?')}x{hdr.get('pixel_size_y','?')}x{hdr.get('pixel_size_x','?')}",
                "injected_dose": hdr.get("injected_dose", ""),
                "dose_units": hdr.get("dose_units", ""),
                "isotope": hdr.get("isotope", ""),
            }
        )
    return rows


def summarize_dcm_files(set3_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[Path, List[RTImageVolume]]]:
    rows = []
    by_dir: Dict[Path, List[RTImageVolume]] = {}
    for dcm_path in sorted(set3_dir.rglob("*.dcm")):
        rel = dcm_path.relative_to(set3_dir).as_posix()
        try:
            vol = load_rt_image_dicom(dcm_path)
            ds = vol.raw_ds
            study_time = str(getattr(ds, "StudyTime", "")).strip() if ds is not None else ""
            series_desc = str(getattr(ds, "SeriesDescription", "")).strip() if ds is not None else ""
            patient_name = str(getattr(ds, "PatientName", "")).strip() if ds is not None else ""
            x_min, x_max = float(vol.x_coords.min()), float(vol.x_coords.max())
            y_min, y_max = float(vol.y_coords.min()), float(vol.y_coords.max())
            z_min, z_max = float(vol.z_coords.min()), float(vol.z_coords.max())
            rows.append(
                {
                    "rel_path": rel,
                    "modality": vol.modality,
                    "units": vol.units,
                    "study_date": vol.study_date,
                    "study_time": study_time,
                    "patient_name": patient_name,
                    "series_desc": series_desc,
                    "shape": vol.shape,
                    "spacing_mm": tuple(round(s, 4) for s in vol.spacing),
                    "origin_mm": tuple(round(float(x), 3) for x in vol.origin),
                    "orientation": np.round(vol.orientation, 3).tolist(),
                    "x_range_mm": (round(x_min, 2), round(x_max, 2)),
                    "y_range_mm": (round(y_min, 2), round(y_max, 2)),
                    "z_range_mm": (round(z_min, 2), round(z_max, 2)),
                    "data_min": round(float(vol.data.min()), 4),
                    "data_max": round(float(vol.data.max()), 4),
                }
            )
            by_dir.setdefault(dcm_path.parent, []).append(vol)
        except Exception as e:
            rows.append({"rel_path": rel, "error": repr(e)})
    return rows, by_dir


def _dominant_plane_label(roi: RTRoi) -> str:
    axes = set()
    for c in roi.contours:
        dom = int(np.argmax(np.abs(c.basis[0])))
        axes.add({0: "X-norm(Sag)", 1: "Y-norm(Trans)", 2: "Z-norm(Cor)"}[dom])
    return ",".join(sorted(axes))


def _quadrant_label(cx: float, cz: float, vol: Optional[RTImageVolume]) -> str:
    """
    Labels approximate position in the transaxial (X, Z) plane relative to volume center.
    In RT_Image transaxial display (extent=[x_max, x_min, z_min, z_max]):
      - Left of display = larger X (more positive / less negative X)
      - Right of display = smaller X (more negative X)
      - Top of display = larger Z (more positive / less negative Z, Dorsal)
      - Bottom of display = smaller Z (more negative Z, Ventral)
    """
    if vol is None:
        return "?"
    x_mid = 0.5 * (float(vol.x_coords.min()) + float(vol.x_coords.max()))
    z_mid = 0.5 * (float(vol.z_coords.min()) + float(vol.z_coords.max()))
    horiz = "Left(X>mid)" if cx >= x_mid else "Right(X<mid)"
    vert = "Top(Z>mid)" if cz >= z_mid else "Bottom(Z<mid)"
    return f"{vert}-{horiz}"


def inspect_roi_files(
    set3_dir: Path, dcm_by_dir: Dict[Path, List[RTImageVolume]]
) -> List[Dict[str, Any]]:
    roi_paths = sorted(
        [p for p in set3_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".roi"]
    )
    results = []
    for roi_path in roi_paths:
        rel = roi_path.relative_to(set3_dir).as_posix()
        entry: Dict[str, Any] = {"rel_path": rel, "size_bytes": roi_path.stat().st_size}
        try:
            ver, rois = load_rt_image_roi(roi_path)
            entry["version"] = ver
            entry["num_rois"] = len(rois)

            # Pick PET volume in same directory if available (prefer modality PT)
            vols = dcm_by_dir.get(roi_path.parent, [])
            pet_vol = next((v for v in vols if v.modality == "PT"), None)
            if pet_vol is None and vols:
                pet_vol = vols[0]

            # Special case: 1. 15 Dec has pet_set_1.dcm and pet_set_2.dcm in the same dir
            if roi_path.name.lower().startswith("pet_set_1"):
                pet_vol = next((v for v in vols if "set_1" in v.path.name.lower()), pet_vol)

            entry["paired_vol"] = pet_vol.path.name if pet_vol is not None else None
            if pet_vol is not None:
                entry["paired_vol_center_mm"] = [
                    round(0.5 * (float(pet_vol.x_coords.min()) + float(pet_vol.x_coords.max())), 2),
                    round(0.5 * (float(pet_vol.y_coords.min()) + float(pet_vol.y_coords.max())), 2),
                    round(0.5 * (float(pet_vol.z_coords.min()) + float(pet_vol.z_coords.max())), 2),
                ]

            roi_summaries = []
            for roi in rois:
                bmin, bmax = roi.bounds_mm
                cent = roi.centroid_mm
                ctypes = sorted({c.roi_type for c in roi.contours})
                r_info: Dict[str, Any] = {
                    "name": roi.name,
                    "num_contours": len(roi.contours),
                    "contour_types": ",".join(ctypes),
                    "plane": _dominant_plane_label(roi),
                    "centroid_xyz_mm": [round(float(x), 2) for x in cent],
                    "x_mm": [round(float(bmin[0]), 2), round(float(bmax[0]), 2)],
                    "y_mm": [round(float(bmin[1]), 2), round(float(bmax[1]), 2)],
                    "z_mm": [round(float(bmin[2]), 2), round(float(bmax[2]), 2)],
                    "quadrant": _quadrant_label(float(cent[0]), float(cent[2]), pet_vol),
                    "orig_path": roi.original_roi_path,
                    "ref_images": roi.ref_images,
                }
                if pet_vol is not None:
                    mask = build_roi_3d_mask(pet_vol, roi)
                    vals = np.asarray(pet_vol.data[mask], dtype=np.float64)
                    dz, dy, dx = pet_vol.spacing
                    vox_cm3 = (dx * dy * dz) / 1000.0
                    r_info["pet_points"] = int(vals.size)
                    r_info["pet_vol_cm3"] = round(vals.size * vox_cm3, 4)
                    if vals.size > 0:
                        r_info["pet_mean"] = round(float(vals.mean()), 4)
                        r_info["pet_max"] = round(float(vals.max()), 4)
                    else:
                        r_info["pet_mean"] = None
                        r_info["pet_max"] = None
                roi_summaries.append(r_info)
            entry["rois"] = roi_summaries
        except Exception as e:
            entry["error"] = repr(e)
        results.append(entry)
    return results


def main() -> None:
    root = Path("/workspace")
    set3_dir = root / "images" / "set3"

    print("=" * 90)
    print("1. RAW SCANNER HEADERS (.img.hdr) IN images/set3")
    print("=" * 90)
    hdr_rows = summarize_hdr_files(set3_dir)
    if hdr_rows:
        df_hdr = pd.DataFrame(hdr_rows)
        print(df_hdr.to_string(index=False))

    print("\n" + "=" * 90)
    print("2. RT_IMAGE DICOM VOLUMES (.dcm) IN images/set3")
    print("=" * 90)
    dcm_rows, dcm_by_dir = summarize_dcm_files(set3_dir)
    for r in dcm_rows:
        print(json.dumps(r))

    print("\n" + "=" * 90)
    print("3. RT_IMAGE ROI FILES (.roi / .ROI) IN images/set3")
    print("=" * 90)
    roi_results = inspect_roi_files(set3_dir, dcm_by_dir)
    for entry in roi_results:
        print(f"\n--- {entry['rel_path']} ({entry['size_bytes']} bytes) ---")
        if "error" in entry:
            print(f"  ERROR: {entry['error']}")
            continue
        print(
            f"  Version: {entry['version']} | ROIs: {entry['num_rois']} | "
            f"Paired vol: {entry['paired_vol']} | Vol center (mm): {entry.get('paired_vol_center_mm')}"
        )
        rows = []
        for r in entry["rois"]:
            rows.append(
                {
                    "ROI Name": r["name"],
                    "Type": r["contour_types"],
                    "N": r["num_contours"],
                    "Plane": r["plane"],
                    "Centroid (X,Y,Z)": f"({r['centroid_xyz_mm'][0]:6.1f}, {r['centroid_xyz_mm'][1]:6.1f}, {r['centroid_xyz_mm'][2]:6.1f})",
                    "X [min,max]": f"[{r['x_mm'][0]:.1f},{r['x_mm'][1]:.1f}]",
                    "Y [min,max]": f"[{r['y_mm'][0]:.1f},{r['y_mm'][1]:.1f}]",
                    "Z [min,max]": f"[{r['z_mm'][0]:.1f},{r['z_mm'][1]:.1f}]",
                    "Quadrant": r["quadrant"],
                    "Pts": r.get("pet_points", ""),
                    "Vol(cc)": r.get("pet_vol_cm3", ""),
                    "Mean": r.get("pet_mean", ""),
                    "Max": r.get("pet_max", ""),
                }
            )
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        # Print unique ref_images / orig_path if informative
        refs = sorted({ref for r in entry["rois"] for ref in r["ref_images"]})
        origs = sorted({r["orig_path"] for r in entry["rois"] if r["orig_path"]})
        if origs:
            print(f"  Original ROI paths: {origs}")
        if refs:
            print(f"  Ref images: {refs}")


if __name__ == "__main__":
    main()
