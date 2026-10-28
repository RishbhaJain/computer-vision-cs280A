"""Homography recovery and two inverse image warps, implemented with NumPy."""
import numpy as np


def homography_system(im1_pts, im2_pts):
    source = np.asarray(im1_pts, dtype=float)
    target = np.asarray(im2_pts, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("The two point arrays must have the same shape (n, 2).")
    if len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("At least four finite point pairs are required.")
    A = np.zeros((2 * len(source), 8))
    for i, ((x, y), (u, v)) in enumerate(zip(source, target)):
        A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
    return A, target.reshape(-1)


def computeH(im1_pts, im2_pts):
    """Return a 3x3 homography mapping im1_pts to im2_pts using all pairs."""
    A, b = homography_system(im1_pts, im2_pts)
    h, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        raise ValueError("The point pairs do not determine a homography.")
    return np.append(h, 1).reshape(3, 3)


def _canvas(im, H):
    height, width = im.shape[:2]
    corners = np.array([[0, 0, 1], [width-1, 0, 1],
                        [width-1, height-1, 1], [0, height-1, 1]], dtype=float)
    q = (H @ corners.T).T
    if q[:, 2].min() <= 0 <= q[:, 2].max():
        raise ValueError("The image crosses projective infinity.")
    q = q[:, :2] / q[:, 2:3]
    xmin, ymin = np.floor(q.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(q.max(axis=0)).astype(int)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    return (int(ymax-ymin+1), int(xmax-xmin+1)), T @ H


def _warp(im, H, method, output_shape=None, return_mask=False):
    im = np.asarray(im)
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("H must be a finite 3x3 matrix.")
    if im.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or color image.")
    if output_shape is None:
        output_shape, H = _canvas(im, H)
    out_h, out_w = map(int, output_shape)
    if min(out_h, out_w) <= 0 or out_h*out_w > 40_000_000:
        raise ValueError("Invalid or excessively large output canvas.")
    h, w = im.shape[:2]
    output = np.zeros((out_h, out_w) + im.shape[2:], dtype=im.dtype)
    mask = np.zeros((out_h, out_w), dtype=bool)
    inverse = np.linalg.inv(H)
    for first in range(0, out_h, 128):
        last = min(first+128, out_h)
        y_out, x_out = np.mgrid[first:last, :out_w]
        denominator = inverse[2, 0]*x_out+inverse[2, 1]*y_out+inverse[2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (inverse[0, 0]*x_out+inverse[0, 1]*y_out+inverse[0, 2])/denominator
            y = (inverse[1, 0]*x_out+inverse[1, 1]*y_out+inverse[1, 2])/denominator
        valid = (np.isfinite(x) & np.isfinite(y) & (x >= -1e-7)
                 & (x <= w-1+1e-7) & (y >= -1e-7) & (y <= h-1+1e-7))
        x = np.clip(np.where(valid, x, 0), 0, w-1)
        y = np.clip(np.where(valid, y, 0), 0, h-1)
        if method == "nearest":
            values = im[np.rint(y).astype(int), np.rint(x).astype(int)].copy()
        else:
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
            dx, dy = x-x0, y-y0
            if im.ndim == 3:
                dx, dy = dx[..., None], dy[..., None]
            values = ((1-dx)*(1-dy)*im[y0, x0] + dx*(1-dy)*im[y0, x1]
                      + (1-dx)*dy*im[y1, x0] + dx*dy*im[y1, x1])
            if np.issubdtype(im.dtype, np.integer):
                limits = np.iinfo(im.dtype)
                values = np.clip(np.rint(values), limits.min, limits.max)
            values = values.astype(im.dtype)
        values[~valid] = 0
        output[first:last] = values
        mask[first:last] = valid
    return (output, mask) if return_mask else output


def warpImageNearestNeighbor(im, H, output_shape=None, return_mask=False):
    return _warp(im, H, "nearest", output_shape, return_mask)


def warpImageBilinear(im, H, output_shape=None, return_mask=False):
    return _warp(im, H, "bilinear", output_shape, return_mask)
