"""Forensic-style ballistics image examination report builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from uuid import uuid4

REPORT_FOOTER = (
    "Automated preliminary digital comparison. Confirm with validated microscopy methods "
    "and a qualified firearms / toolmark examiner before operational or legal use."
)

MATCH_THRESHOLD = 72.0
INCONCLUSIVE_LOW = 48.0


@dataclass(frozen=True)
class ReportInputs:
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
    orb_visual_available: bool


@dataclass(frozen=True)
class ForensicVerdict:
    code: str  # MATCH | NO_MATCH | INCONCLUSIVE
    label: str
    summary: str
    css_class: str


def overall_similarity_percent(ssim: float, hist_corr: float, orb_score: float) -> float:
    raw = 0.50 * ssim + 0.35 * hist_corr + 0.15 * orb_score
    return float(max(0.0, min(100.0, round(raw * 100.0, 1))))


def derive_verdict(data: ReportInputs) -> ForensicVerdict:
    p = data.match_percent
    if p >= MATCH_THRESHOLD and data.ssim >= 0.55:
        return ForensicVerdict(
            code="MATCH",
            label="MATCH",
            summary=(
                "Based on combined striation-surface metrics, specimen B exhibits sufficient "
                "correspondence with specimen A to support a preliminary **match** classification "
                f"({p:.1f}% overall similarity index)."
            ),
            css_class="ft-verdict-match",
        )
    if p < INCONCLUSIVE_LOW or data.ssim < 0.40:
        return ForensicVerdict(
            code="NO_MATCH",
            label="NO MATCH",
            summary=(
                "Quantitative and visual metrics indicate insufficient correspondence between "
                f"specimens ({p:.1f}% similarity index). Preliminary classification: **no match**."
            ),
            css_class="ft-verdict-nomatch",
        )
    return ForensicVerdict(
        code="INCONCLUSIVE",
        label="INCONCLUSIVE",
        summary=(
            f"Similarity index {p:.1f}% falls in the borderline range. Insufficient evidence for "
            "a definitive match or exclusion without additional views, scale alignment, or "
            "microscopic examination."
        ),
        css_class="ft-verdict-inconclusive",
    )


def similarity_bullets(data: ReportInputs) -> list[str]:
    items: list[str] = []
    if data.ssim >= 0.80:
        items.append(
            f"High structural correspondence (SSIM {data.ssim:.3f}) in striation / surface layout after normalization."
        )
    elif data.ssim >= 0.55:
        items.append(
            f"Moderate structural correspondence (SSIM {data.ssim:.3f}) in overall pattern and edge layout."
        )
    else:
        items.append(f"Limited structural correspondence (SSIM {data.ssim:.3f}).")

    if data.hist_corr >= 0.85:
        items.append(
            f"Brightness / tonal distribution closely aligned (histogram correlation {data.hist_corr:.3f})."
        )
    elif data.hist_corr >= 0.60:
        items.append(
            f"Partial tonal alignment (histogram correlation {data.hist_corr:.3f})."
        )

    if data.orb_score >= 0.08:
        items.append(
            f"Localized toolmark / texture features show agreement (ORB ratio {data.orb_score:.3f})."
        )
    elif data.orb_visual_available:
        items.append("Feature endpoints detected; sparse ORB agreement on available texture.")

    bright_delta = abs(data.mean_brightness_a - data.mean_brightness_b)
    if bright_delta < 8:
        items.append("Exposure and overall luminance are comparable between specimens.")

    if not items:
        items.append("No significant automated similarities detected above reporting thresholds.")
    return items


def difference_bullets(data: ReportInputs) -> list[str]:
    items: list[str] = []
    items.append(
        f"Mean grayscale separation: {data.mean_pixel_diff:.1f}/255; "
        f"{data.diff_area_percent:.1f}% of aligned pixels exceed a 15-level difference threshold."
    )
    bright_delta = abs(data.mean_brightness_a - data.mean_brightness_b)
    if bright_delta >= 8:
        brighter = "Specimen A" if data.mean_brightness_a > data.mean_brightness_b else "Specimen B"
        items.append(f"{brighter} is brighter by approximately {bright_delta:.1f} gray levels (lighting / exposure variance).")

    if data.ssim < 0.55:
        items.append("Low SSIM indicates misalignment, different capture angle, or non-corresponding toolmark regions.")
    if data.hist_corr < 0.60:
        items.append("Histogram divergence suggests different lighting, oxidation, or non-comparable surfaces.")
    if data.orb_score < 0.05:
        items.append(
            "Minimal ORB correspondence — common with smooth primer metal or insufficient macro texture."
        )
    if not data.orb_visual_available:
        items.append("Insufficient micro-texture for reliable automated feature-line charting (ORB).")

    if data.width_a != data.width_b or data.height_a != data.height_b:
        items.append(
            f"Original resolutions differ (A: {data.width_a}×{data.height_a}, B: {data.width_b}×{data.height_b}); "
            "comparison uses rescaled alignment only."
        )
    return items


def _list_html(items: list[str]) -> str:
    return "".join(f"<li>{escape(t)}</li>" for t in items)


def build_screen_summary(data: ReportInputs, expert_name: str) -> str:
    verdict = derive_verdict(data)
    sim = similarity_bullets(data)
    diff = difference_bullets(data)
    expert = escape(expert_name.strip() or "—")
    return f"""
<div class="ft-final-result">
  <p class="ft-expert-line"><strong>Expert examiner:</strong> {expert}</p>
  <div class="ft-verdict-banner {verdict.css_class}">
    <span class="ft-verdict-title">Forensic conclusion</span>
    <span class="ft-verdict-label">{escape(verdict.label)}</span>
    <span class="ft-sim-index">Similarity index: {data.match_percent:.1f}%</span>
  </div>
  <p class="ft-conclusion-text">{verdict.summary.replace('**', '')}</p>

  <div class="ft-two-col">
    <div class="ft-sim-box">
      <h3 class="ft-report-h3">Similarities</h3>
      <ul class="ft-diff-list ft-sim-list">{_list_html(sim)}</ul>
    </div>
    <div class="ft-diff-box">
      <h3 class="ft-report-h3">Differences</h3>
      <ul class="ft-diff-list">{_list_html(diff)}</ul>
    </div>
  </div>
</div>
"""


def build_printable_document(
    data: ReportInputs,
    title: str = "Forensic Ballistics — Toolmark / Cartridge Image Examination Report",
    expert_name: str = "Aryan",
    project_author: str = "Aryan",
) -> str:
    verdict = derive_verdict(data)
    sim = similarity_bullets(data)
    diff = difference_bullets(data)
    report_id = f"FB-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    conclusion = (
        f"**{verdict.label}** — {verdict.summary.replace('**', '')}"
        if verdict.label != "INCONCLUSIVE"
        else f"**INCONCLUSIVE** — {verdict.summary}"
    )
    expert = expert_name.strip() or project_author

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: "Times New Roman", Georgia, serif; margin: 2cm; color: #111; line-height: 1.5; font-size: 11pt; }}
    h1 {{ font-size: 16pt; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 2px solid #111; padding-bottom: 0.35rem; }}
    h2 {{ font-size: 12pt; margin-top: 1.2rem; }}
    .meta {{ font-size: 10pt; color: #333; }}
    .verdict {{ font-size: 14pt; font-weight: bold; margin: 1rem 0; padding: 0.6rem; border: 2px solid #111; text-align: center; }}
    .match {{ border-color: #166534; color: #166534; }}
    .nomatch {{ border-color: #991b1b; color: #991b1b; }}
    .inconclusive {{ border-color: #b45309; color: #b45309; }}
    table {{ width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 10pt; }}
    th, td {{ border: 1px solid #666; padding: 0.35rem 0.5rem; }}
    th {{ background: #eee; }}
    ul {{ margin: 0.35rem 0 0.75rem 1.1rem; }}
    .expert-box {{ border: 1px solid #333; padding: 0.6rem 0.75rem; margin: 1rem 0; background: #fafafa; }}
    .sign {{ margin-top: 2rem; }}
    .footer {{ margin-top: 1.5rem; font-size: 9pt; color: #555; border-top: 1px solid #ccc; padding-top: 0.5rem; }}
    @media print {{ body {{ margin: 1.5cm; }} }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="meta"><strong>Project:</strong> {escape(project_author)} · <strong>Report ID:</strong> {escape(report_id)} · <strong>Date:</strong> {escape(now)}<br />
  <strong>Specimen A (image):</strong> {data.width_a}×{data.height_a} px ·
  <strong>Specimen B (image):</strong> {data.width_b}×{data.height_b} px</p>

  <div class="expert-box">
    <strong>Forensic expert examiner:</strong> {escape(expert)}<br />
    <strong>Examination type:</strong> Digital toolmark / cartridge image comparison
  </div>

  <h2>I. Examination objective</h2>
  <p>Compare two digital images of cartridge-case / toolmark-related surfaces to assess whether
  automated similarity metrics support correspondence (match) or separation (no match).</p>

  <h2>II. Methodology</h2>
  <p>Digital normalization (resize, pad), grayscale SSIM, intensity-histogram correlation,
  ORB feature correspondence, pixel-difference mapping, and blend overlay at {data.blend * 100:.0f}% B-opacity.</p>

  <h2>III. Quantitative analysis</h2>
  <table>
    <tr><th>Parameter</th><th>Result</th></tr>
    <tr><td>Overall similarity index</td><td>{data.match_percent:.1f}%</td></tr>
    <tr><td>Structural index (SSIM)</td><td>{data.ssim:.4f} ({data.ssim * 100:.1f}%)</td></tr>
    <tr><td>Histogram correlation</td><td>{data.hist_corr:.4f} ({data.hist_corr * 100:.1f}%)</td></tr>
    <tr><td>Toolmark feature ratio (ORB)</td><td>{data.orb_score:.4f} ({data.orb_score * 100:.1f}%)</td></tr>
    <tr><td>Mean pixel difference</td><td>{data.mean_pixel_diff:.2f} / 255</td></tr>
    <tr><td>High-difference area</td><td>{data.diff_area_percent:.1f}% of image</td></tr>
  </table>

  <h2>IV. Observed similarities</h2>
  <ul>{_list_html(sim)}</ul>

  <h2>V. Observed differences</h2>
  <ul>{_list_html(diff)}</ul>

  <h2>VI. Forensic examination conclusion</h2>
  <div class="verdict {'match' if verdict.code == 'MATCH' else 'nomatch' if verdict.code == 'NO_MATCH' else 'inconclusive'}">
    {escape(verdict.label)} · Similarity index {data.match_percent:.1f}%
  </div>
  <p>{escape(conclusion.replace('**', ''))}</p>

  <h2>VII. Expert certification</h2>
  <p class="sign">I, <strong>{escape(expert)}</strong>, certify that the above forensic ballistics image
  examination was performed in accordance with this automated analysis record and that the
  stated conclusion reflects the reported metrics and observations.</p>
  <p><strong>Examiner:</strong> {escape(expert)} &nbsp;&nbsp; <strong>Date:</strong> {escape(now.split(' UTC')[0])}</p>

  <p class="footer">{escape(REPORT_FOOTER)}</p>
</body>
</html>
"""


def write_report_html(
    data: ReportInputs,
    out_dir: Path,
    expert_name: str = "Aryan",
    project_author: str = "Aryan",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"forensic_examination_{stamp}.html"
    path.write_text(
        build_printable_document(data, expert_name=expert_name, project_author=project_author),
        encoding="utf-8",
    )
    return path


# Backwards compatibility for tests / imports
educational_match_percent = overall_similarity_percent