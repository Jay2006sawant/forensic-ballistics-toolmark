"""Image comparison logic (no Gradio — safe for unit tests)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from report_builder import ReportInputs, overall_similarity_percent

PAD_COLOR = (248, 250, 252)


class CompareError(Exception):
    """User-visible comparison failure."""


@dataclass(frozen=True)
class ComparisonResult:
    ssim: float
    hist_corr: float
    orb_score: float
    match_percent: float
    mean_pixel_diff: float
    diff_area_percent: float
    mean_brightness_a: float
    mean_brightness_b: float
    width_a: int
    height_a: int
    width_b: int
    height_b: int
    blend: float
    pil_a: Image.Image
    pil_b: Image.Image
    pil_overlay: Image.Image
    pil_diff: Image.Image
    pil_matches: Image.Image | None

    def report_inputs(self) -> ReportInputs:
        return ReportInputs(
            ssim=self.ssim,
            hist_corr=self.hist_corr,
            orb_score=self.orb_score,
            match_percent=self.match_percent,
            mean_pixel_diff=self.mean_pixel_diff,
            diff_area_percent=self.diff_area_percent,
            mean_brightness_a=self.mean_brightness_a,
            mean_brightness_b=self.mean_brightness_b,
            width_a=self.width_a,
            height_a=self.height_a,
            width_b=self.width_b,
            height_b=self.height_b,
            blend=self.blend,
            orb_visual_available=self.pil_matches is not None,
        )


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    try:
        arr = np.array(img.convert("RGB"))
    except Exception as e:
        raise CompareError(f"Could not read image: {e}") from e
    if arr.size == 0:
        raise CompareError("Image is empty or unreadable.")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def align_size(a: np.ndarray, b: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if ha < 2 or wa < 2 or hb < 2 or wb < 2:
        raise CompareError("Each image must be at least 2×2 pixels.")
    scale = min(max_side / max(ha, wa), max_side / max(hb, wb), 1.0)
    na = cv2.resize(a, (int(wa * scale), int(ha * scale)), interpolation=cv2.INTER_AREA)
    nb = cv2.resize(b, (int(wb * scale), int(hb * scale)), interpolation=cv2.INTER_AREA)
    h = max(na.shape[0], nb.shape[0])
    w = max(na.shape[1], nb.shape[1])

    def pad(im: np.ndarray) -> np.ndarray:
        ph, pw = h - im.shape[0], w - im.shape[1]
        return cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=PAD_COLOR)

    return pad(na), pad(nb)


def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.var(), y.var()
    sig_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2)
    if den == 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def _diff_stats(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float]:
    diff = cv2.absdiff(gray_a, gray_b)
    mean_diff = float(diff.mean())
    diff_area_percent = float((diff > 15).mean() * 100.0)
    return mean_diff, diff_area_percent


def orb_match_score(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, np.ndarray | None]:
    try:
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray_a, None)
        kp2, des2 = orb.detectAndCompute(gray_b, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0.0, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        score = len(good) / max(len(kp1), len(kp2), 1)
        score = float(np.clip(score, 0.0, 1.0))
        vis = cv2.drawMatches(
            gray_a, kp1, gray_b, kp2, good[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        if vis.ndim == 2:
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis_bgr = vis
        return score, vis_bgr
    except cv2.error as e:
        raise CompareError(f"Feature matching failed: {e}") from e


def run_comparison(image_a: Image.Image, image_b: Image.Image, blend: float) -> ComparisonResult:
    width_a, height_a = image_a.size
    width_b, height_b = image_b.size

    a = pil_to_bgr(image_a)
    b = pil_to_bgr(image_b)
    a, b = align_size(a, b)

    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    ssim = ssim_gray(gray_a, gray_b)
    orb_score, match_vis = orb_match_score(gray_a, gray_b)
    mean_diff, diff_area = _diff_stats(gray_a, gray_b)
    mean_bright_a = float(gray_a.mean())
    mean_bright_b = float(gray_b.mean())

    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_corr = float(np.clip(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL), 0.0, 1.0))

    diff = cv2.absdiff(gray_a, gray_b)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)
    alpha = float(np.clip(blend, 0.0, 1.0))
    overlay = cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)

    match_pil = bgr_to_pil(match_vis) if match_vis is not None else None
    match_percent = overall_similarity_percent(ssim, hist_corr, orb_score)

    return ComparisonResult(
        ssim=ssim,
        hist_corr=hist_corr,
        orb_score=orb_score,
        match_percent=match_percent,
        mean_pixel_diff=mean_diff,
        diff_area_percent=diff_area,
        mean_brightness_a=mean_bright_a,
        mean_brightness_b=mean_bright_b,
        width_a=width_a,
        height_a=height_a,
        width_b=width_b,
        height_b=height_b,
        blend=alpha,
        pil_a=bgr_to_pil(a),
        pil_b=bgr_to_pil(b),
        pil_overlay=bgr_to_pil(overlay),
        pil_diff=bgr_to_pil(diff_color),
        pil_matches=match_pil,
    )