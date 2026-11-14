"""Forensic ballistics image examination prototype."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from compare_core import CompareError, run_comparison
from report_builder import build_screen_summary, derive_verdict, write_report_html

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
PROJECT_AUTHOR = "Aryan"
HF_SPACE_USER = "aryan"
CUSTOM_CSS = (ROOT / "styles.css").read_text(encoding="utf-8")

PRINT_HELP = """
<div class="ft-print-hint">
  <strong>Print:</strong> Download the HTML report, open it in your browser, then
  <kbd>Ctrl+P</kbd> / <kbd>Cmd+P</kbd> → Save as PDF or print.
</div>
"""



def _build_theme() -> gr.themes.Soft:
    """Teal + cream palette; dark-mode tokens mirror light (no OS dark UI)."""
    light = dict(
        body_background_fill="#f7f5f2",
        background_fill_primary="#ffffff",
        background_fill_secondary="#fbfaf8",
        block_background_fill="#ffffff",
        block_border_color="#e3ddd4",
        block_border_width="1px",
        block_label_background_fill="#ccfbf1",
        block_label_text_color="#115e59",
        body_text_color="#2c3338",
        input_background_fill="#ffffff",
        input_border_color="#e3ddd4",
        border_color_primary="#e3ddd4",
        button_primary_background_fill="#0d9488",
        button_primary_background_fill_hover="#0f766e",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_text_color="#2c3338",
        slider_color="#0d9488",
        shadow_drop="0 2px 10px rgba(44, 51, 56, 0.07)",
    )
    dark_keys = (
        "body_background_fill",
        "background_fill_primary",
        "background_fill_secondary",
        "block_background_fill",
        "block_border_color",
        "block_label_background_fill",
        "block_label_text_color",
        "body_text_color",
        "input_background_fill",
        "input_border_color",
        "border_color_primary",
        "button_primary_background_fill",
        "button_primary_background_fill_hover",
        "button_secondary_background_fill",
        "button_secondary_text_color",
        "slider_color",
    )
    mirrored_dark = {f"{key}_dark": light[key] for key in dark_keys}
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        **light,
        **mirrored_dark,
        block_title_background_fill="#ccfbf1",
        block_title_background_fill_dark="#ccfbf1",
        block_title_text_color="#115e59",
        block_title_text_color_dark="#115e59",
    )


def _metrics_html(ssim: float, hist_corr: float, orb_score: float, match_percent: float) -> str:
    return f"""
<div class="ft-metrics">
  <p class="ft-section-label" style="margin-top:0">Detailed metrics</p>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
        <th>As %</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Overall similarity index</strong></td>
        <td class="ft-value">{match_percent:.1f}%</td>
        <td>Ballistics composite score</td>
      </tr>
      <tr>
        <td>Grayscale SSIM</td>
        <td class="ft-value">{ssim:.3f}</td>
        <td>{ssim * 100:.1f}%</td>
      </tr>
      <tr>
        <td>Histogram correlation</td>
        <td class="ft-value">{hist_corr:.3f}</td>
        <td>{hist_corr * 100:.1f}%</td>
      </tr>
      <tr>
        <td>Toolmark feature ratio (ORB)</td>
        <td class="ft-value">{orb_score:.3f}</td>
        <td>{orb_score * 100:.1f}%</td>
      </tr>
    </tbody>
  </table>
</div>
"""


def _status_html(kind: str, title: str, detail: str) -> str:
    return f"""
<div class="ft-status ft-status-{kind}" role="alert">
  <strong>{title}</strong>
  <p>{detail}</p>
</div>
"""


def _fail(
    status_msg: str,
) -> tuple[str, str, str, None, None, None, None, None, None, None]:
    return (status_msg, "", "", None, None, None, None, None, None, None)


def _expert_display(name: str | None) -> str:
    text = (name or "").strip()
    return text if text else PROJECT_AUTHOR


def compare_images(
    image_a: Image.Image | None,
    image_b: Image.Image | None,
    blend: float,
    expert_name: str,
) -> tuple[
    str,
    str,
    str,
    str | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    str | None,
]:
    if image_a is None and image_b is None:
        return _fail(
            _status_html("info", "Waiting for specimens", "Upload Specimen A and Specimen B, then run forensic analysis.")
        )

    if image_a is None or image_b is None:
        missing = "Image A" if image_a is None else "Image B"
        return _fail(
            _status_html(
                "warn",
                "Missing upload",
                f"Please upload <strong>{missing}</strong> before comparing.",
            )
        )

    try:
        result = run_comparison(image_a, image_b, blend)
    except CompareError as e:
        return _fail(_status_html("error", "Comparison failed", str(e)))
    except Exception as e:
        return _fail(
            _status_html(
                "error",
                "Unexpected error",
                f"{type(e).__name__}: {e} — try different images (JPG/PNG, at least 32×32 px).",
            )
        )

    inputs = result.report_inputs()
    expert = _expert_display(expert_name)
    verdict = derive_verdict(inputs)
    status = _status_html(
        "ok",
        "Examination complete",
        f"Forensic conclusion: <strong>{verdict.label}</strong> "
        f"(similarity index {result.match_percent:.1f}%). "
        f"Expert: <strong>{expert}</strong>. Download the formal report below.",
    )
    summary = build_screen_summary(inputs, expert)
    metrics = _metrics_html(result.ssim, result.hist_corr, result.orb_score, result.match_percent)
    printable = f'<div id="ft-print-root" class="ft-print-root">{summary}{metrics}</div>'
    report_path = write_report_html(inputs, REPORTS_DIR, expert_name=expert, project_author=PROJECT_AUTHOR)
    report_download = str(report_path)

    return (
        status,
        summary,
        metrics,
        printable,
        result.pil_a,
        result.pil_b,
        result.pil_overlay,
        result.pil_diff,
        result.pil_matches,
        report_download,
    )


def build_ui() -> gr.Blocks:
    hero = f"""
<div class="ft-hero">
  <div class="ft-badge">Ballistics examination</div>
  <h1>Forensic cartridge &amp; toolmark analysis</h1>
  <p>Upload specimen images to obtain a match / no match conclusion, similarity and
  difference findings, and a printable forensic examination report.</p>
  <p class="ft-author">Forensic ballistics project — <strong>{PROJECT_AUTHOR}</strong></p>
</div>
"""

    with gr.Blocks(
        title="Forensic Ballistics Image Examination",
        theme=_build_theme(),
        css=CUSTOM_CSS,
        head='<meta name="color-scheme" content="light only" />',
    ) as demo:
        gr.HTML(hero)

        gr.HTML('<p class="ft-section-label">Step 1 — Specimen images</p>')
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="ft-panel"):
                img_a = gr.Image(
                    type="pil",
                    label="Specimen A",
                    sources=["upload", "clipboard"],
                    height=280,
                )
            with gr.Column(elem_classes="ft-panel"):
                img_b = gr.Image(
                    type="pil",
                    label="Specimen B",
                    sources=["upload", "clipboard"],
                    height=280,
                )

        gr.HTML('<p class="ft-section-label">Step 2 — Examiner details</p>')
        with gr.Row():
            expert_box = gr.Textbox(
                label="Name of expert examiner",
                placeholder="e.g. Aryan Kumar — Firearms & Toolmark Examiner",
                value=PROJECT_AUTHOR,
                info="This name appears on the forensic report and expert certification.",
            )

        gr.HTML('<p class="ft-section-label">Step 3 — Adjust overlay</p>')
        with gr.Row():
            blend = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.05,
                label="Blend opacity for Image B",
                info="Slide to preview how the two images stack.",
            )
        btn = gr.Button("Run forensic analysis", variant="primary", size="lg")
        status = gr.HTML(value=_status_html("info", "Ready", "Upload specimens and run analysis."))

        gr.HTML('<p class="ft-section-label">Step 4 — Forensic conclusion &amp; examination report</p>')
        final_summary = gr.HTML()
        report_metrics = gr.HTML()
        printable_area = gr.HTML()
        gr.HTML(PRINT_HELP)
        report_file = gr.File(label="Download forensic examination report (HTML / print to PDF)", interactive=False)

        gr.HTML('<p class="ft-section-label">Step 5 — Metrics &amp; visuals</p>')
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="ft-panel"):
                out_a = gr.Image(label="Normalized A", height=240, show_label=True)
            with gr.Column(elem_classes="ft-panel"):
                out_b = gr.Image(label="Normalized B", height=240, show_label=True)
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="ft-panel"):
                out_overlay = gr.Image(label="Blend overlay", height=240)
            with gr.Column(elem_classes="ft-panel"):
                out_diff = gr.Image(label="Difference map", height=240)
        with gr.Column(elem_classes="ft-panel"):
            out_matches = gr.Image(
                label="Feature matches (ORB)",
                height=280,
                show_label=True,
            )

        outputs = [
            status,
            final_summary,
            report_metrics,
            printable_area,
            out_a,
            out_b,
            out_overlay,
            out_diff,
            out_matches,
            report_file,
        ]
        inputs = [img_a, img_b, blend, expert_box]
        btn.click(compare_images, inputs=inputs, outputs=outputs, show_progress=True)
        blend.release(compare_images, inputs=inputs, outputs=outputs, show_progress=False)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    on_hf = bool(os.getenv("SPACE_ID") or os.getenv("SYSTEM") == "spaces")
    ui.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=not on_hf,
        show_error=True,
        show_api=False,
    )