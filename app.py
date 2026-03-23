import hashlib
import html
import math
import os
from functools import lru_cache
from urllib.parse import quote

import gradio as gr

from clip_search import search_by_image, search_by_text

IMAGE_DIR = "images"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATASET_LIMIT = 200


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_all_images(limit=DATASET_LIMIT):
    paths = [
        os.path.join(root, fname)
        for root, _, files in os.walk(IMAGE_DIR)
        for fname in files
        if os.path.splitext(fname)[1].lower() in EXTS
    ]
    return sorted(paths)[:limit]


def category_from_path(path):
    return os.path.basename(os.path.dirname(path)) or "dataset"


def label_from_result(path, score=None, prefix="score"):
    cat = category_from_path(path).title()
    return cat if score is None else f"{cat} · {prefix}: {score:.4f}"


def gallery_items(paths, score_map=None, prefix="score"):
    return [
        (p, label_from_result(p, None if score_map is None else score_map.get(p), prefix))
        for p in paths if os.path.exists(p)
    ]


def status_html(msg, kind="info"):
    colors = {"ok": "#22c55e", "warn": "#f59e0b", "info": "#94a3b8", "err": "#ef4444"}
    c = colors.get(kind, "#94a3b8")
    icons = {"ok": "✓", "warn": "⚠", "info": "◎", "err": "✕"}
    icon = icons.get(kind, "◎")
    esc = html.escape(msg)
    return (
        f'<span class="status-pill status-{kind}">'
        f'<span class="status-icon">{icon}</span>{esc}</span>'
    )


@lru_cache(maxsize=1024)
def build_file_url(path):
    return f"/gradio_api/file={quote(os.path.abspath(path))}"


# ---------------------------------------------------------------------------
# Build payload
# ---------------------------------------------------------------------------

def build_payload(result_paths, score_map=None, prefix="score", msg="", kind="info"):
    dataset_paths = get_all_images(DATASET_LIMIT)
    score_map = score_map or {}
    return (
        gallery_items(result_paths, score_map, prefix),
        status_html(msg, kind),
    )


# ---------------------------------------------------------------------------
# Search handlers
# ---------------------------------------------------------------------------

def explore_dataset():
    paths = get_all_images(DATASET_LIMIT)
    if not paths:
        return (
            [],
            status_html("No images found in dataset", "err"),
        )
    return build_payload(paths,
                         msg=f"{len(paths)} images loaded", kind="info")


def run_text_search(query, k):
    query = (query or "").strip()
    paths = get_all_images(DATASET_LIMIT)
    if not query:
        return ([], status_html("Enter a search query", "warn"))
    try:
        results = search_by_text(query, k=int(k))
        valid = [(p, s) for p, s in results if os.path.exists(p)]
        if not valid:
            return ([], status_html(f'No results for "{query}"', "warn"))
        rp = [p for p, _ in valid]
        sm = {p: s for p, s in valid}
        return build_payload(rp, sm, "score",
                             msg=f'{len(rp)} results for "{query}"', kind="ok")
    except Exception as exc:
        return ([], status_html(f"Error: {exc}", "err"))


def run_image_search(image, k):
    paths = get_all_images(DATASET_LIMIT)
    if image is None:
        return ([], status_html("Upload an image to search", "warn"))
    tmp = "_query_tmp.jpg"
    try:
        image.save(tmp)
        results = search_by_image(tmp, k=int(k))
        valid = [(p, s) for p, s in results if os.path.exists(p)]
        if not valid:
            return ([], status_html("No similar images found", "warn"))
        rp = [p for p, _ in valid]
        sm = {p: s for p, s in valid}
        return build_payload(rp, sm, "similarity",
                             msg=f"{len(rp)} visually similar images", kind="ok")
    except Exception as exc:
        return ([], status_html(f"Error: {exc}", "err"))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def toggle_view(mode):
    show_grid = mode == "Grid"
    return gr.update(visible=show_grid), gr.update(visible=not show_grid)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f8f6f2;
  --surf: #ffffff;
  --surf2: #f2ede6;
  --surf3: #e8e1d8;
  --border: rgba(60,40,20,0.12);
  --text: #1a1208;
  --muted: #8a7a6a;
  --accent: #c84b31;
  --accent2: #e8682f;
  --sh1: 0 2px 8px rgba(0,0,0,.06);
  --sh2: 0 8px 32px rgba(0,0,0,.10);
  --sh3: 0 20px 64px rgba(0,0,0,.14);
  --r: 20px;
  --rm: 14px;
  --rs: 10px;
  --ease: .22s cubic-bezier(.22,1,.36,1);
  --font: 'Syne', sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

[data-theme="dark"] {
  --bg: #0d0f12;
  --surf: #161a20;
  --surf2: #1e2330;
  --surf3: #262d3a;
  --border: rgba(180,200,230,0.10);
  --text: #eef2f8;
  --muted: #7a8899;
  --accent: #ff6b4a;
  --accent2: #ffaa44;
  --sh1: 0 2px 8px rgba(0,0,0,.3);
  --sh2: 0 8px 32px rgba(0,0,0,.4);
  --sh3: 0 20px 64px rgba(0,0,0,.5);
}

html, body { min-height: 100%; }

.gradio-container {
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  background: var(--bg) !important;
  font-family: var(--font) !important;
  color: var(--text) !important;
  transition: background var(--ease), color var(--ease);
}

footer, .footer, .svelte-1ipelgc { display: none !important; }
.gr-box, .gr-form, .gr-panel { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── App shell ── */
#app { padding: 20px; display: flex; flex-direction: column; gap: 16px; }

/* ── Topbar ── */
#topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 16px 22px;
  background: var(--surf); border: 1px solid var(--border);
  border-radius: calc(var(--r) + 4px);
  box-shadow: var(--sh1);
}

.brand { display: flex; flex-direction: column; gap: 3px; }
.brand-k { font: 500 0.68rem var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
.brand-t { font-size: 1.6rem; font-weight: 800; letter-spacing: -.05em; line-height: 1; color: #E3C79F;}
.brand-t em { font-style: normal; color: var(--accent); }
.brand-sub { font-size: .88rem; color: var(--muted); }

/* Theme switcher */
.theme-sw {
  display: inline-flex; gap: 4px; padding: 4px;
  border: 1px solid var(--border); border-radius: 999px;
  background: var(--surf2);
}
.theme-sw button {
  border: none; cursor: pointer; padding: 8px 16px; border-radius: 999px;
  font: 600 0.8rem var(--font); background: transparent; color: var(--muted);
  transition: all var(--ease);
}
.theme-sw button:hover { color: var(--text); }
.theme-sw button.active {
  background: var(--surf); color: var(--text); box-shadow: var(--sh1);
}

/* ── Body layout ── */
#body { display: grid; grid-template-columns: 300px minmax(0,1fr); gap: 16px; align-items: start; }

/* ── Sidebar ── */
#sidebar {
  position: sticky; top: 20px;
  background: var(--surf); border: 1px solid var(--border);
  border-radius: var(--r); box-shadow: var(--sh1);
  padding: 20px; display: flex; flex-direction: column; gap: 16px;
}

.sid-title { font-size: 1rem; font-weight: 700; letter-spacing: -.03em; }
.sid-desc { font-size: .86rem; color: var(--muted); line-height: 1.5; }
.sid-label { font: 500 0.67rem var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.sid-hr { height: 1px; background: var(--border); }

/* View toggle */
.view-toggle .wrap {
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  background: var(--surf2) !important;
  padding: 4px !important;
  gap: 4px !important;
}
.view-toggle label {
  border-radius: 999px !important; border: none !important;
  background: transparent !important; color: var(--muted) !important;
  padding: 8px 14px !important; font: 600 .8rem var(--font) !important;
  text-transform: none !important; letter-spacing: 0 !important;
  transition: all var(--ease) !important; margin: 0 !important;
}
.view-toggle label:hover { color: var(--text) !important; }
.view-toggle label.selected {
  background: var(--surf) !important; color: var(--text) !important;
  box-shadow: var(--sh1) !important;
}

/* Inputs */
textarea, input[type="text"] {
  background: var(--surf2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--rm) !important; color: var(--text) !important;
  font-family: var(--font) !important; font-size: .92rem !important;
  padding: 12px 14px !important;
  transition: border-color var(--ease), box-shadow var(--ease) !important;
  resize: vertical !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(200,75,49,.15) !important;
  outline: none !important;
}
textarea::placeholder, input::placeholder { color: var(--muted) !important; }
label, .gr-form label {
  color: var(--muted) !important; font: 500 0.67rem var(--mono) !important;
  letter-spacing: .14em !important; text-transform: uppercase !important;
}

/* Upload box */
#img-upload {
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--rm) !important;
  background: var(--surf2) !important;
  transition: border-color var(--ease) !important;
  min-height: 180px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
#img-upload:hover {
  border-color: var(--accent) !important;
}
#img-upload .wrap {
  border: none !important;
  background: transparent !important;
  width: 100% !important;
  min-height: 180px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
#img-upload .wrap .icon-wrap {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 12px !important;
}
#img-upload .wrap .icon-wrap svg {
  width: 48px !important;
  height: 48px !important;
  stroke: var(--muted) !important;
}
#img-upload .wrap .upload-text {
  font-size: .82rem !important;
  color: var(--text) !important;   
  font-weight: 500 !important;
  text-align: center !important;
  margin-top: 8px !important;
}
#img-upload img {
  max-height: 160px !important;
  object-fit: contain !important;
  border-radius: var(--rs) !important;
}

/* Buttons */
.btn-p, .btn-s {
  width: 100% !important; border-radius: 999px !important;
  padding: 11px 16px !important; font: 700 .88rem var(--font) !important;
  letter-spacing: -.01em !important; cursor: pointer !important;
  transition: transform var(--ease), box-shadow var(--ease) !important;
}
.btn-p {
  border: none !important;
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #fff !important; box-shadow: 0 8px 24px rgba(200,75,49,.3) !important;
}
.btn-p:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 30px rgba(200,75,49,.4) !important; }
.btn-p:active { transform: scale(.97) !important; }
.btn-s {
  border: 1px solid var(--border) !important; background: var(--surf2) !important;
  color: var(--text) !important;
}
.btn-s:hover { border-color: var(--accent) !important; color: var(--accent) !important; transform: translateY(-1px) !important; }

input[type="range"] { accent-color: var(--accent) !important; }

/* ── Content ── */
#content { display: flex; flex-direction: column; gap: 12px; }

/* Status bar */
#status-bar {
  padding: 12px 18px; background: var(--surf); border: 1px solid var(--border);
  border-radius: var(--rm); box-shadow: var(--sh1); min-height: 44px;
  display: flex; align-items: center;
}

.status-pill {
  display: inline-flex; align-items: center; gap: 8px;
  font: 500 .78rem var(--mono); letter-spacing: .04em;
}
.status-icon { font-size: .9em; }
.status-ok { color: #22c55e; }
.status-warn { color: #f59e0b; }
.status-info { color: var(--muted); }
.status-err { color: #ef4444; }

/* Result shell */
.result-shell {
  background: var(--surf); border: 1px solid var(--border);
  border-radius: var(--r); box-shadow: var(--sh1); overflow: hidden;
}

.result-head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 18px 20px 10px;
}
.result-head h2 { font-size: 1.1rem; font-weight: 700; letter-spacing: -.04em; }
.result-head p { font-size: .84rem; color: var(--muted); margin-top: 2px; }
.result-badge {
  padding: 6px 12px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--surf2); color: var(--muted); font: 500 .68rem var(--mono);
  letter-spacing: .08em; text-transform: uppercase; white-space: nowrap;
}

/* ── Gallery ── */
#gallery-wrap { padding: 0 16px 16px; }

/* Gallery grid - full width with proper spacing */
#main-gal {
  width: 100% !important;
}
#main-gal .grid-wrap,
#main-gal > div {
  display: block !important;
  width: 100% !important;
}
#main-gal > div > div {
  display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)) !important;
  gap: 16px !important;
  width: 100% !important;
}

#main-gal .thumbnail-item {
  border: none !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  background: var(--surf2) !important;
  box-shadow: var(--sh1) !important;
  aspect-ratio: 1 / 1 !important;
  width: 100% !important;
  transition: transform var(--ease), box-shadow var(--ease) !important;
}
#main-gal .thumbnail-item:hover {
  transform: translateY(-3px) scale(1.02) !important;
  box-shadow: var(--sh3) !important;
}
#main-gal .thumbnail-item img {
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
}

/* ── Responsive ── */
@media (max-width: 1100px) {
  #body { grid-template-columns: 1fr !important; }
  #sidebar { position: relative; top: 0; }
}
@media (max-width: 720px) {
  #app { padding: 12px; }
  #topbar { flex-direction: column; align-items: flex-start; }
  #main-gal > div > div { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)) !important; }
}
"""

# ---------------------------------------------------------------------------
# JavaScript — theme switching
# ---------------------------------------------------------------------------

JS = """
function applyTheme(t) {
  if (t === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }

  localStorage.setItem("ps-theme", t);

  document.querySelectorAll(".tsw-btn").forEach(function(b) {
    b.classList.toggle("active", b.dataset.t === t);
  });
}

function handleThemeClick(e) {
  const btn = e.target.closest(".tsw-btn");
  if (!btn) return;

  e.preventDefault();
  applyTheme(btn.dataset.t);
}

function initTheme() {
  const saved = localStorage.getItem("ps-theme") || "light";
  applyTheme(saved);

  document.body.removeEventListener("click", handleThemeClick);
  document.body.addEventListener("click", handleThemeClick);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTheme);
} else {
  initTheme();
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, js=JS, title="PixelSeek", theme=gr.themes.Base()) as demo:
    with gr.Column(elem_id="app"):

        # Topbar
        gr.HTML("""
        <div id="topbar">
          <div class="brand">
            <div class="brand-k">Visual Search Engine</div>
            <h1 class="brand-t">Pixel<em>Seek</em></h1>
            <p class="brand-sub">Find images by text or visual similarity.</p>
          </div>
          <div class="theme-sw" aria-label="Theme">
            <button class="tsw-btn active" data-t="light" type="button">☀ Light</button>
            <button class="tsw-btn" data-t="dark" type="button">☾ Dark</button>
          </div>
        </div>
        """)

        with gr.Row(elem_id="body"):

            # Sidebar
            with gr.Column(elem_id="sidebar", scale=0, min_width=300):
                gr.HTML("<h2 class='sid-title'>Search controls</h2>"
                        "<p class='sid-desc'>Search by text, upload a reference image, or explore the full dataset.</p>")


                gr.HTML('<div class="sid-hr"></div>')
                gr.HTML('<div class="sid-label">Text search</div>')
                txt_input = gr.Textbox(
                    placeholder="quiet forest trail, neon cityscape, ceramic bowl…",
                    show_label=False, lines=2,
                )
                txt_btn = gr.Button("Search by text", elem_classes="btn-p")

                gr.HTML('<div class="sid-hr"></div>')
                gr.HTML('<div class="sid-label">Image reference</div>')
                img_input = gr.Image(
                    type="pil", show_label=False,
                    elem_id="img-upload", height=180,
                )
                img_btn = gr.Button("Search by image", elem_classes="btn-p")

                gr.HTML('<div class="sid-hr"></div>')
                k_slider = gr.Slider(minimum=1, maximum=20, value=9, step=1, label="Top-k results")

                gr.HTML('<div class="sid-hr"></div>')
                explore_btn = gr.Button("Explore Dataset", elem_classes="btn-s")

            # Content
            with gr.Column(elem_id="content", scale=1):

                # Status
                with gr.Column(elem_id="status-bar"):
                    status_bar = gr.HTML(
                        status_html("Ready — explore the dataset or run a search.", "info")
                    )

                # Grid view
                with gr.Column(elem_classes="result-shell", visible=True) as grid_shell:
                    gr.HTML("""
                    <div class="result-head">
                      <div>
                        <h2>Grid results</h2>
                        <p>Visual browsing with ranked matches.</p>
                      </div>
                      <div class="result-badge">Gallery</div>
                    </div>""")
                    with gr.Column(elem_id="gallery-wrap"):
                        gallery = gr.Gallery(
                            show_label=False,
                            columns=4,
                            rows=3,
                            height="auto",
                            object_fit="cover",
                            elem_id="main-gal",
                            allow_preview=True,
                        )

                # Semantic map view (placeholder - functionality removed)
                with gr.Column(elem_classes="result-shell", visible=False) as map_shell:
                    gr.HTML("""
                    <div class="result-head">
                      <div>
                        <h2>Semantic Map</h2>
                        <p>Feature visualization (coming soon).</p>
                      </div>
                      <div class="result-badge">Map</div>
                    </div>
                    <div style="padding: 40px; text-align: center; color: var(--muted);">
                      Semantic map feature has been removed. Please use Grid view.
                    </div>
                    """)

    # ── Wire up events ──
    initial_gallery, initial_status = explore_dataset()

    demo.load(
        fn=lambda: (initial_gallery, initial_status),
        outputs=[gallery, status_bar],
    )

    txt_btn.click(fn=run_text_search, inputs=[txt_input, k_slider], outputs=[gallery, status_bar])
    txt_input.submit(fn=run_text_search, inputs=[txt_input, k_slider], outputs=[gallery, status_bar])
    img_btn.click(fn=run_image_search, inputs=[img_input, k_slider], outputs=[gallery, status_bar])
    explore_btn.click(fn=explore_dataset, outputs=[gallery, status_bar])


if __name__ == "__main__":
    demo.launch()