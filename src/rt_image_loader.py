"""
Loader, cropper, and analysis utilities for RT_Image (IDL) DICOM 3D volumes,
Siemens Inveon (.img / .hdr) volumes, and binary RT_Image .ROI files.
"""

from __future__ import annotations

import datetime
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid
from skimage.draw import polygon


@dataclass
class RTImageVolume:
    """
    Represents a 3D image volume (from RT_Image DICOM or Siemens Inveon .img/.hdr).

    Coordinate convention in RT_Image Set 1 / Set 2 (ImageOrientationPatient = [[-1,0,0],[0,0,-1]]):
      - Axis 0 (k, planes/slices, size nz, spacing dz): physical Y coordinate
            Y(k) = origin[1] - k * dz   <=>   k = (origin[1] - Y) / dz
      - Axis 1 (j, rows, size ny, spacing dy): physical Z coordinate
            Z(j) = origin[2] - j * dy   <=>   j = (origin[2] - Z) / dy
      - Axis 2 (i, cols, size nx, spacing dx): physical X coordinate
            X(i) = origin[0] - i * dx   <=>   i = (origin[0] - X) / dx
    """
    path: Path
    modality: str
    units: str
    study_date: str
    shape: Tuple[int, int, int]  # (nz, ny, nx) = (planes, rows, cols)
    spacing: Tuple[float, float, float]  # (dz, dy, dx) in mm
    origin: np.ndarray  # ImagePositionPatient (3,) = [ox, oy, oz] in mm
    orientation: np.ndarray  # ImageOrientationPatient (2, 3): [row_cos, col_cos]
    rescale_slope: float
    rescale_intercept: float
    data: np.ndarray  # 3D array (nz, ny, nx) in physical units
    raw_ds: Optional[pydicom.Dataset] = field(default=None, repr=False)

    @property
    def x_coords(self) -> np.ndarray:
        """Physical X coordinates (mm) for column indices i = 0 .. nx-1 (decreasing X)."""
        return self.origin[0] - np.arange(self.shape[2]) * self.spacing[2]

    @property
    def y_coords(self) -> np.ndarray:
        """Physical Y coordinates (mm) for plane indices k = 0 .. nz-1 (decreasing Y)."""
        return self.origin[1] - np.arange(self.shape[0]) * self.spacing[0]

    @property
    def z_coords(self) -> np.ndarray:
        """Physical Z coordinates (mm) for row indices j = 0 .. ny-1 (decreasing Z)."""
        return self.origin[2] - np.arange(self.shape[1]) * self.spacing[1]

    def world_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        """
        Maps physical (x, y, z) coordinates in mm (shape (..., 3))
        to continuous voxel indices (k, j, i) = (plane, row, col).
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        dz, dy, dx = self.spacing
        ox, oy, oz = self.origin
        i = (ox - xyz[..., 0]) / dx
        k = (oy - xyz[..., 1]) / dz
        j = (oz - xyz[..., 2]) / dy
        return np.stack([k, j, i], axis=-1)


@dataclass
class RTContour:
    """Represents a single 2D planar contour (freehand polygon, ellipse, or rectangle) within an ROI."""
    roi_type: str  # 'freehand', 'ellipse', or 'rectangle'
    vertices: np.ndarray  # (N, 3) float64 physical coordinates (x, y, z) in mm
    center: np.ndarray  # (3,) float64 physical center (x, y, z) in mm
    basis: np.ndarray  # (3, 3) float64 [normal, u_axis, v_axis]
    thickness: float  # slice thickness in mm
    flags: bytes  # 5-byte trailer flags
    ellipse_diameters: Optional[Tuple[float, float]] = None  # (dx, dy) in mm if ellipse/rectangle


@dataclass
class RTRoi:
    """Represents a 3D Region of Interest (collection of 2D planar contours) from RT_Image."""
    name: str
    original_roi_path: str
    color_rgba: Tuple[int, int, int, int]
    line_thickness: float
    opacity: float
    ref_images: List[str]
    contours: List[RTContour]
    raw_bytes: bytes = field(default=b"", repr=False)

    @property
    def all_vertices(self) -> np.ndarray:
        if not self.contours:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack([c.vertices for c in self.contours])

    @property
    def bounds_mm(self) -> Tuple[np.ndarray, np.ndarray]:
        pts = self.all_vertices
        if len(pts) == 0:
            return np.zeros(3), np.zeros(3)
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def centroid_mm(self) -> np.ndarray:
        pts = self.all_vertices
        if len(pts) == 0:
            return np.zeros(3)
        return pts.mean(axis=0)


def load_rt_image_dicom(path: str | Path) -> RTImageVolume:
    """
    Loads an RT_Image headerless Implicit VR Little Endian 3D DICOM volume.
    Handles RT_Image quirks such as missing 128-byte DICM preamble and float strings in NumberOfFrames.
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = pydicom.dcmread(str(path), force=True)

    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.is_implicit_VR = True
    ds.is_little_endian = True

    rows = int(ds.Rows)
    cols = int(ds.Columns)

    if (0x0028, 0x0012) in ds:
        planes = int(ds[(0x0028, 0x0012)].value)
    elif hasattr(ds, "NumberOfFrames"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            planes = int(float(str(ds[(0x0028, 0x0008)].value).strip()))
    elif (0x0054, 0x0081) in ds:
        planes = int(ds[(0x0054, 0x0081)].value)
    else:
        planes = 1

    if (0x0028, 0x0008) in ds:
        ds[(0x0028, 0x0008)].value = str(planes)

    bits_allocated = int(getattr(ds, "BitsAllocated", 16))
    pixel_rep = int(getattr(ds, "PixelRepresentation", 1))
    if bits_allocated == 16:
        dtype = np.int16 if pixel_rep == 1 else np.uint16
    elif bits_allocated == 32:
        dtype = np.int32 if pixel_rep == 1 else np.uint32
    elif bits_allocated == 8:
        dtype = np.int8 if pixel_rep == 1 else np.uint8
    else:
        raise ValueError(f"Unsupported BitsAllocated={bits_allocated}")

    raw_bytes = ds.PixelData
    expected_count = planes * rows * cols
    raw_arr = np.frombuffer(raw_bytes, dtype=dtype, count=expected_count).reshape((planes, rows, cols))

    slope = float(str(getattr(ds, "RescaleSlope", 1.0)).strip())
    intercept = float(str(getattr(ds, "RescaleIntercept", 0.0)).strip())
    data = raw_arr.astype(np.float32) * slope + intercept

    px_spacing = [float(x) for x in ds.PixelSpacing]
    dy, dx = px_spacing[0], px_spacing[1]
    if hasattr(ds, "SpacingBetweenSlices"):
        dz = float(str(ds.SpacingBetweenSlices).strip())
    elif hasattr(ds, "SliceThickness"):
        dz = float(str(ds.SliceThickness).strip())
    else:
        dz = 1.0

    origin = np.array([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)
    orient = np.array([float(x) for x in ds.ImageOrientationPatient], dtype=np.float64).reshape(2, 3)

    modality = str(getattr(ds, "Modality", "OT")).strip()
    units = ""
    if hasattr(ds, "RescaleType") and str(ds.RescaleType).strip():
        units = str(ds.RescaleType).strip()
    elif (0x0054, 0x1001) in ds:
        units = str(ds[(0x0054, 0x1001)].value).strip()
    elif modality == "CT":
        units = "HU"

    raw_date = str(getattr(ds, "StudyDate", "")).strip()
    if len(raw_date) == 8 and raw_date.isdigit():
        study_date = f"{raw_date[4:6]}/{raw_date[6:8]}/{raw_date[0:4]}"
    else:
        study_date = raw_date

    return RTImageVolume(
        path=path,
        modality=modality,
        units=units,
        study_date=study_date,
        shape=(planes, rows, cols),
        spacing=(dz, dy, dx),
        origin=origin,
        orientation=orient,
        rescale_slope=slope,
        rescale_intercept=intercept,
        data=data,
        raw_ds=ds,
    )


def parse_inveon_hdr(hdr_path: str | Path) -> Dict[str, str]:
    """Parses a Siemens Inveon ASCII .hdr file into a dictionary of key-value strings."""
    hdr_path = Path(hdr_path)
    params: Dict[str, str] = {}
    with hdr_path.open("r", encoding="latin1", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            # Keep first occurrence of top-level keys unless frame-level scale_factor/minimum/maximum
            if key not in params or key in ("scale_factor", "minimum", "maximum"):
                params[key] = val
    return params


def load_inveon_volume(
    img_path: str | Path,
    hdr_path: Optional[str | Path] = None,
    mmap: bool = True,
) -> RTImageVolume:
    """
    Loads a raw Siemens Inveon .img + .hdr volume (such as Set 2's i100118_v1.ct.img or .pet.img).
    Reverses the axial slice order ([::-1, :, :]) to match RT_Image's physical coordinate convention.
    """
    img_path = Path(img_path)
    if hdr_path is None:
        candidate = Path(str(img_path) + ".hdr")
        if candidate.exists():
            hdr_path = candidate
        else:
            raise FileNotFoundError(f"Could not find .hdr file for {img_path}")
    hdr = parse_inveon_hdr(hdr_path)

    nx = int(hdr["x_dimension"])
    ny = int(hdr["y_dimension"])
    nz = int(hdr["z_dimension"])

    dx = float(hdr["pixel_size_x"])
    dy = float(hdr["pixel_size_y"])
    dz = float(hdr["pixel_size_z"])

    data_type = int(hdr.get("data_type", "2"))
    dtype_map = {
        1: "<i1",
        2: "<i2",
        3: "<i4",
        4: "<f4",
        5: ">f4",
        6: ">i2",
        7: ">i4",
    }
    if data_type not in dtype_map:
        raise ValueError(f"Unsupported Inveon data_type={data_type} in {hdr_path}")
    dtype = dtype_map[data_type]

    scale_factor = float(hdr.get("scale_factor", "1.0"))
    modality_code = int(hdr.get("modality", "1"))
    modality = "PT" if modality_code == 0 else ("CT" if modality_code == 1 else "OT")
    units = "HU" if modality == "CT" else ("counts" if modality == "PT" else "")

    study_date = "06/02/2011"
    if "scan_time" in hdr:
        try:
            dt = datetime.datetime.strptime(hdr["scan_time"], "%a %b %d %H:%M:%S %Y")
            study_date = dt.strftime("%m/%d/%Y")
        except ValueError:
            pass

    if mmap:
        raw_arr = np.memmap(img_path, dtype=dtype, mode="r", shape=(nz, ny, nx))
    else:
        raw_arr = np.fromfile(img_path, dtype=dtype, count=nz * ny * nx).reshape((nz, ny, nx))

    # Reverse axial slice index k ([::-1, :, :]) to align with RT_Image's coordinate system
    oriented_arr = raw_arr[::-1, :, :]
    if abs(scale_factor - 1.0) > 1e-9:
        data = np.asarray(oriented_arr, dtype=np.float32) * scale_factor
    else:
        data = oriented_arr

    return RTImageVolume(
        path=img_path,
        modality=modality,
        units=units,
        study_date=study_date,
        shape=(nz, ny, nx),
        spacing=(dz, dy, dx),
        origin=np.zeros(3, dtype=np.float64),
        orientation=np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        rescale_slope=scale_factor,
        rescale_intercept=0.0,
        data=data,
        raw_ds=None,
    )


def load_volume_auto(path: str | Path) -> RTImageVolume:
    """Automatically loads either an RT_Image DICOM (.dcm) or a Siemens Inveon (.img/.hdr) volume."""
    path = Path(path)
    if path.name.lower().endswith(".hdr"):
        img_path = Path(str(path)[:-4])
        return load_inveon_volume(img_path, hdr_path=path)
    elif path.name.lower().endswith(".img"):
        return load_inveon_volume(path)
    else:
        return load_rt_image_dicom(path)


def _read_str(buf: bytes, offset: int) -> Tuple[str, int]:
    """Reads a uint32-prefixed ASCII string from buf at offset."""
    (length,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    s = buf[offset : offset + length].decode("latin1", errors="replace")
    return s, offset + length


def load_rt_image_roi(path: str | Path, num_ellipse_pts: int = 360) -> Tuple[str, List[RTRoi]]:
    """
    Parses an RT_Image binary .ROI file (e.g. 'version 3.10   07/09/2008').
    Returns (version_header, list_of_rois).
    """
    path = Path(path)
    buf = path.read_bytes()

    version_header = buf[:25].decode("latin1").strip()
    offset = 25

    (num_rois,) = struct.unpack_from("<I", buf, offset)
    offset += 4

    rois: List[RTRoi] = []
    for _ in range(num_rois):
        roi_start_offset = offset
        name, offset = _read_str(buf, offset)
        (num_contours,) = struct.unpack_from("<H", buf, offset)
        offset += 2

        orig_roi_path, offset = _read_str(buf, offset)

        r, g, b, a, flag1, flag2, line_thick, opacity, num_refs = struct.unpack_from(
            "<BBBBHHffH", buf, offset
        )
        offset += struct.calcsize("<BBBBHHffH")

        ref_images: List[str] = []
        for _ in range(num_refs):
            ref_img, offset = _read_str(buf, offset)
            ref_images.append(ref_img)

        contours: List[RTContour] = []
        for _ in range(num_contours):
            roi_type, offset = _read_str(buf, offset)
            roi_type_clean = roi_type.strip().lower()

            if roi_type_clean == "freehand":
                (n_pts,) = struct.unpack_from("<I", buf, offset)
                offset += 4
                verts = np.frombuffer(buf, dtype="<f8", count=n_pts * 3, offset=offset).reshape(n_pts, 3).copy()
                offset += n_pts * 3 * 8

                center = np.frombuffer(buf, dtype="<f8", count=3, offset=offset).copy()
                offset += 24
                basis = np.frombuffer(buf, dtype="<f8", count=9, offset=offset).reshape(3, 3).copy()
                offset += 72
                (thickness,) = struct.unpack_from("<d", buf, offset)
                offset += 8
                flags = buf[offset : offset + 5]
                offset += 5

                contours.append(
                    RTContour(
                        roi_type="freehand",
                        vertices=verts,
                        center=center,
                        basis=basis,
                        thickness=thickness,
                        flags=flags,
                    )
                )
            elif roi_type_clean == "ellipse":
                center = np.frombuffer(buf, dtype="<f8", count=3, offset=offset).copy()
                offset += 24
                dx, dy = struct.unpack_from("<dd", buf, offset)
                offset += 16
                basis = np.frombuffer(buf, dtype="<f8", count=9, offset=offset).reshape(3, 3).copy()
                offset += 72
                (thickness,) = struct.unpack_from("<d", buf, offset)
                offset += 8
                flags = buf[offset : offset + 5]
                offset += 5

                theta = np.linspace(0.0, 2.0 * np.pi, num_ellipse_pts, endpoint=False)
                u_vec = basis[1]
                v_vec = basis[2]
                rx, ry = dx / 2.0, dy / 2.0
                verts = (
                    center[None, :]
                    + (rx * np.cos(theta))[:, None] * u_vec[None, :]
                    + (ry * np.sin(theta))[:, None] * v_vec[None, :]
                )

                contours.append(
                    RTContour(
                        roi_type="ellipse",
                        vertices=verts,
                        center=center,
                        basis=basis,
                        thickness=thickness,
                        flags=flags,
                        ellipse_diameters=(dx, dy),
                    )
                )
            elif roi_type_clean == "rectangle":
                center = np.frombuffer(buf, dtype="<f8", count=3, offset=offset).copy()
                offset += 24
                dx, dy = struct.unpack_from("<dd", buf, offset)
                offset += 16
                basis = np.frombuffer(buf, dtype="<f8", count=9, offset=offset).reshape(3, 3).copy()
                offset += 72
                (thickness,) = struct.unpack_from("<d", buf, offset)
                offset += 8
                flags = buf[offset : offset + 5]
                offset += 5

                u_vec = basis[1]
                v_vec = basis[2]
                hx, hy = dx / 2.0, dy / 2.0
                corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
                verts = (
                    center[None, :]
                    + corners[:, 0:1] * u_vec[None, :]
                    + corners[:, 1:2] * v_vec[None, :]
                )

                contours.append(
                    RTContour(
                        roi_type="rectangle",
                        vertices=verts,
                        center=center,
                        basis=basis,
                        thickness=thickness,
                        flags=flags,
                        ellipse_diameters=(dx, dy),
                    )
                )
            else:
                raise ValueError(f"Unsupported RT_Image contour type '{roi_type}' at byte offset {offset}")

        roi_end_offset = offset
        rois.append(
            RTRoi(
                name=name,
                original_roi_path=orig_roi_path,
                color_rgba=(r, g, b, a),
                line_thickness=line_thick,
                opacity=opacity,
                ref_images=ref_images,
                contours=contours,
                raw_bytes=buf[roi_start_offset:roi_end_offset],
            )
        )

    return version_header, rois


def save_rt_image_roi_subset(
    version_header: str,
    rois: Sequence[RTRoi],
    out_path: str | Path,
) -> Path:
    """
    Writes a subset of RTRoi objects to a binary RT_Image .ROI file (preserving exact original contour bytes).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr_bytes = version_header.encode("latin1")[:25].ljust(25, b" ")
    count_bytes = struct.pack("<I", len(rois))
    body_bytes = b"".join(r.raw_bytes for r in rois)
    out_path.write_bytes(hdr_bytes + count_bytes + body_bytes)
    return out_path


def build_roi_3d_mask(vol: RTImageVolume, roi: RTRoi) -> np.ndarray:
    """
    Rasterizes an RTRoi onto the 3D voxel grid (nz, ny, nx) of an RTImageVolume.
    Automatically detects which orthogonal plane each contour was drawn on from `c.basis[0]` (the plane normal):
      - Normal along Y (|basis[0, 1]| > 0.5): contour in (X, Z) plane, extruded along Y (axis 0 = k)
      - Normal along X (|basis[0, 0]| > 0.5): contour in (Y, Z) plane, extruded along X (axis 2 = i)
      - Normal along Z (|basis[0, 2]| > 0.5): contour in (X, Y) plane, extruded along Z (axis 1 = j)
    """
    dz, dy, dx = vol.spacing
    nz, ny, nx = vol.shape
    ox, oy, oz = vol.origin

    mask = np.zeros((nz, ny, nx), dtype=bool)
    for c in roi.contours:
        normal = np.abs(c.basis[0])
        dom_axis = int(np.argmax(normal))

        if dom_axis == 1:
            y0 = c.center[1]
            k_center = (oy - y0) / dz
            half_slab = (c.thickness / 2.0) / dz
            k_min = max(0, int(np.ceil(k_center - half_slab - 1e-4)))
            k_max = min(nz - 1, int(np.floor(k_center + half_slab + 1e-4)))
            ks = range(k_min, k_max + 1) if k_max >= k_min else [int(round(k_center))]

            ic = (ox - c.vertices[:, 0]) / dx
            jc = (oz - c.vertices[:, 2]) / dy
            rr, cc = polygon(jc, ic, shape=(ny, nx))
            for k in ks:
                if 0 <= k < nz:
                    mask[k, rr, cc] = True

        elif dom_axis == 0:
            x0 = c.center[0]
            i_center = (ox - x0) / dx
            half_slab = (c.thickness / 2.0) / dx
            i_min = max(0, int(np.ceil(i_center - half_slab - 1e-4)))
            i_max = min(nx - 1, int(np.floor(i_center + half_slab + 1e-4)))
            iis = range(i_min, i_max + 1) if i_max >= i_min else [int(round(i_center))]

            kc = (oy - c.vertices[:, 1]) / dz
            jc = (oz - c.vertices[:, 2]) / dy
            rr, cc = polygon(kc, jc, shape=(nz, ny))
            for i in iis:
                if 0 <= i < nx:
                    mask[rr, cc, i] = True

        else:
            z0 = c.center[2]
            j_center = (oz - z0) / dy
            half_slab = (c.thickness / 2.0) / dy
            j_min = max(0, int(np.ceil(j_center - half_slab - 1e-4)))
            j_max = min(ny - 1, int(np.floor(j_center + half_slab + 1e-4)))
            jjs = range(j_min, j_max + 1) if j_max >= j_min else [int(round(j_center))]

            kc = (oy - c.vertices[:, 1]) / dz
            ic = (ox - c.vertices[:, 0]) / dx
            rr, cc = polygon(kc, ic, shape=(nz, nx))
            for j in jjs:
                if 0 <= j < ny:
                    mask[rr, j, cc] = True

    return mask


def crop_volume_to_body_roi(vol: RTImageVolume, body_roi: RTRoi) -> RTImageVolume:
    """
    Crops a 3D RTImageVolume to the exact voxel bounding box of a rectangular `body_roi`
    (such as Set 2's 'c8 M0 body', 'c8 M1R body', 'c8 M1L body', 'c8 M1L1R body'),
    updating `origin` ([ox, oy, oz]) so that all physical ROI coordinates remain 100% aligned.
    """
    mask = build_roi_3d_mask(vol, body_roi)
    ks, js, iis = np.where(mask)
    if len(ks) == 0:
        raise ValueError(f"Body ROI '{body_roi.name}' does not intersect volume {vol.path}")

    k_min, k_max = int(ks.min()), int(ks.max())
    j_min, j_max = int(js.min()), int(js.max())
    i_min, i_max = int(iis.min()), int(iis.max())

    dz, dy, dx = vol.spacing
    ox, oy, oz = vol.origin

    new_origin = np.array(
        [
            ox - i_min * dx,
            oy - k_min * dz,
            oz - j_min * dy,
        ],
        dtype=np.float64,
    )

    cropped_data = np.asarray(
        vol.data[k_min : k_max + 1, j_min : j_max + 1, i_min : i_max + 1],
        dtype=np.float32,
    ).copy()

    return RTImageVolume(
        path=vol.path,
        modality=vol.modality,
        units=vol.units,
        study_date=vol.study_date,
        shape=cropped_data.shape,
        spacing=vol.spacing,
        origin=new_origin,
        orientation=vol.orientation.copy(),
        rescale_slope=vol.rescale_slope,
        rescale_intercept=vol.rescale_intercept,
        data=cropped_data,
        raw_ds=None,
    )


def save_rt_image_dicom(vol: RTImageVolume, out_path: str | Path) -> Path:
    """
    Saves an RTImageVolume to disk as an RT_Image-compatible 3D DICOM file
    (matching the format of Set 1's c1_M0_PET_mouse.dcm and c1_M0_CT_mouse.dcm).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = vol.shape
    dz, dy, dx = vol.spacing

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.is_implicit_VR = True
    ds.is_little_endian = True

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.ImageType = ["ORIGINAL", "SECONDARY", "AXIAL"]
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.128" if vol.modality == "PT" else "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    # Format study_date MM/DD/YYYY -> YYYYMMDD if possible
    if "/" in vol.study_date and len(vol.study_date) == 10:
        mm, dd, yyyy = vol.study_date.split("/")
        ds.StudyDate = f"{yyyy}{mm}{dd}"
    else:
        ds.StudyDate = "20110602"

    ds.Modality = vol.modality
    ds.InstitutionName = "Stanford SCi3"
    ds.PatientName = out_path.stem
    ds.PatientID = out_path.stem

    ds.SliceThickness = f"{dz:.8f}"
    ds.SpacingBetweenSlices = f"{dz:.8f}"
    ds.ImagePositionPatient = [f"{x:.6f}" for x in vol.origin]
    ds.ImageOrientationPatient = [f"{x:.7f}" for x in vol.orientation.ravel()]

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = str(nz)
    ds.Rows = int(ny)
    ds.Columns = int(nx)
    ds.add_new((0x0028, 0x0012), "US", int(nz))  # Planes (0028,0012)
    ds.PixelSpacing = [f"{dy:.8f}", f"{dx:.8f}"]

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    # Encode float32 data into int16 with optimal RescaleSlope & RescaleIntercept
    vmin = float(vol.data.min())
    vmax = float(vol.data.max())
    if vol.modality == "CT" and abs(vol.rescale_slope - 1.0) < 1e-6 and abs(vol.rescale_intercept) < 1e-6:
        slope = 1.0
        intercept = 0.0
        raw_int16 = np.clip(np.rint(vol.data), -32768, 32767).astype("<i2")
    else:
        if vmax > vmin:
            slope = (vmax - vmin) / 65534.0
            intercept = vmin + 32768.0 * slope
            raw_int16 = np.clip(np.rint((vol.data - intercept) / slope), -32768, 32767).astype("<i2")
        else:
            slope = 1.0
            intercept = vmin
            raw_int16 = np.zeros(vol.shape, dtype="<i2")

    ds.RescaleIntercept = f"{intercept:.6f}"
    ds.RescaleSlope = f"{slope:.10f}"
    if vol.units:
        ds.RescaleType = vol.units

    ds.PixelData = raw_int16.tobytes()

    # Write headerless Implicit VR Little Endian dataset (matching RT_Image export format)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pydicom.dcmwrite(str(out_path), ds, write_like_original=True)

    return out_path


def compute_roi_statistics(volumes: List[RTImageVolume], rois: List[RTRoi]) -> pd.DataFrame:
    """
    Computes RT_Image-compatible ROI summary statistics across one or more volumes (e.g. PET and CT).
    Columns match RT_Image stats export:
    ROI Label, Image Label, Image Units, Points, Volume, Minimum, Maximum, Mean, Std Dev, Sum
    """
    rows = []
    for vol in volumes:
        dz, dy, dx = vol.spacing
        voxel_vol_cm3 = (dx * dy * dz) / 1000.0
        img_label = f"Unknown {vol.modality} {vol.study_date}".strip()

        for roi in rois:
            mask = build_roi_3d_mask(vol, roi)
            vals = np.asarray(vol.data[mask], dtype=np.float64)
            n_pts = int(vals.size)
            vol_cm3 = n_pts * voxel_vol_cm3
            if n_pts > 0:
                vmin = float(vals.min())
                vmax = float(vals.max())
                vmean = float(vals.mean())
                vstd = float(vals.std(ddof=1)) if n_pts > 1 else 0.0
                vsum = vmean * vol_cm3
            else:
                vmin = vmax = vmean = vstd = vsum = np.nan

            rows.append(
                {
                    "ROI Label": roi.name,
                    "Modality": vol.modality,
                    "Image Label": img_label,
                    "Image Units": vol.units,
                    "Points": n_pts,
                    "Volume (cm3)": round(vol_cm3, 4),
                    "Minimum": round(vmin, 4),
                    "Maximum": round(vmax, 4),
                    "Mean": round(vmean, 4),
                    "Std Dev": round(vstd, 4),
                    "Sum": round(vsum, 5),
                }
            )
    return pd.DataFrame(rows)
