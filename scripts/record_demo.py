"""Record the two dashboard demo clips used in the video (optional tool, needs Playwright and ffmpeg).

    streamlit run app/dashboard.py --server.headless true --server.port 8501   # in another terminal
    python scripts/record_demo.py

Writes docs/demo_1_live_replay.mp4 (the live replay on a real held-out patient) and
docs/demo_2_clinic_tour.mp4 (the synthetic clinic list, patient page, dose scenarios, evidence and data
formats), plus docs/demo_cues.json with the time of each on-screen step, so narration can follow the clips.
Each step is scripted, so a rerun shows the same states. A grey dot marks the pointer and a caption bar
says what each step shows.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8501")
W, H = 1920, 1080

OVERLAY = """
(() => {
  const install = () => {
    if (document.getElementById('demo-dot')) return;
    const dot = document.createElement('div');
    dot.id = 'demo-dot';
    Object.assign(dot.style, {position: 'fixed', width: '22px', height: '22px', borderRadius: '50%',
      background: 'rgba(40,40,40,0.35)', border: '2px solid rgba(255,255,255,0.9)', zIndex: 2147483647,
      pointerEvents: 'none', left: '-50px', top: '-50px', transform: 'translate(-50%, -50%)',
      transition: 'transform 0.12s'});
    const cap = document.createElement('div');
    cap.id = 'demo-cap';
    Object.assign(cap.style, {position: 'fixed', left: '50%', bottom: '26px', transform: 'translateX(-50%)',
      background: 'rgba(15,39,71,0.92)', color: '#fff', font: '600 25px "DejaVu Sans", Arial, sans-serif',
      padding: '14px 26px', borderRadius: '12px', zIndex: 2147483646, pointerEvents: 'none', maxWidth: '1700px',
      textAlign: 'center', lineHeight: '1.35', display: 'none'});
    document.body.appendChild(dot);
    document.body.appendChild(cap);
    document.addEventListener('mousemove', e => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => { dot.style.transform = 'translate(-50%, -50%) scale(0.7)'; }, true);
    document.addEventListener('mouseup', () => { dot.style.transform = 'translate(-50%, -50%)'; }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
})();
"""


class Clip:
    def __init__(self, page, name):
        self.page, self.name, self.t0, self.cues = page, name, time.monotonic(), []

    def caption(self, text):
        self.page.evaluate("t => { const c = document.getElementById('demo-cap'); if (!c) return;"
                           " c.textContent = t; c.style.display = t ? 'block' : 'none'; }", text)
        if text:
            self.cues.append({"t": round(time.monotonic() - self.t0, 1), "caption": text})

    def ready(self, extra=1.0):
        self.page.wait_for_selector("text=Santulan", timeout=120000)
        time.sleep(0.4)
        for _ in range(360):
            if self.page.locator("[data-testid='stStatusWidget']").count() == 0:
                break
            time.sleep(0.25)
        time.sleep(extra)

    def glide(self, locator):
        box = locator.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=18)
        time.sleep(0.25)
        return x, y

    def click(self, locator, extra=1.0):
        locator.scroll_into_view_if_needed()
        self.glide(locator)
        locator.click()
        self.ready(extra)

    def scroll(self, dy, steps=12):
        self.page.mouse.move(W * 0.55, H * 0.55, steps=6)
        for _ in range(steps):
            self.page.mouse.wheel(0, dy / steps)
            time.sleep(0.04)
        time.sleep(0.6)

    def scroll_top(self):
        self.scroll(-6000, steps=10)

    def hold(self, seconds):
        time.sleep(seconds)


def button(page, name):
    return page.get_by_role("button", name=name, exact=True)


def check(page, text):
    return page.get_by_text(text, exact=False).first


def replay_clip(clip):
    """One missed low and one caught low in the same held-out period, then the failure and context paths.

    Readings quoted in the captions are the replay's own outputs for period 2094_0_20211109 (fold 4,
    threshold 79.0 mg/dL chosen on that fold's training patients)."""
    page = clip.page
    clip.caption("Live replay: a real ShanghaiT2DM patient, scored by a model trained without them")
    clip.hold(7)
    clip.scroll(380)
    clip.caption("The clock starts 75 minutes before this period's first low; only records up to the clock are used")
    clip.hold(7)
    clip.caption("Each step adds the next readings and recomputes the forecasts with the evaluation's code")
    for _ in range(3):
        clip.click(button(page, "+15 min"), 2.4)
    clip.click(button(page, "+15 min"), 2.0)
    clip.caption("08:22: the reading is 79.5 mg/dL, just above this fold's 79 threshold, so no alert")
    clip.hold(6)
    clip.click(button(page, "+15 min"), 2.0)
    clip.caption("08:37: 67 mg/dL. The low began without a warning, so the evaluation counts it as missed")
    clip.hold(8)
    jump = page.locator("[data-testid='stSelectbox']").filter(has_text="Jump to 75 minutes before")
    clip.click(jump.get_by_role("combobox"), 1.0)
    page.keyboard.type("21 Nov 03:52", delay=90)
    time.sleep(1.2)
    page.keyboard.press("Enter")
    clip.ready(2.0)
    clip.caption("Jump to 75 minutes before a night-time low in the same period: 02:37, reading 79.5")
    clip.hold(6)
    clip.click(button(page, "+15 min"), 2.0)
    clip.caption("02:52 at night: 76 mg/dL is at or below 79, so the rule alerts. The forecasts expected a rise")
    clip.hold(9)
    clip.click(check(page, "Reveal what happened next"), 2)
    clip.caption("Revealed for scoring only, never given to the model: the reading kept falling")
    clip.hold(6)
    clip.scroll(560)
    clip.caption("The low began at 03:52, so the evaluation scored this alert as matched, 60 minutes ahead")
    clip.hold(8)
    clip.caption("Parity: the live forecast equals the evaluation's stored prediction for this moment")
    clip.hold(7)
    clip.scroll(-560)
    clip.click(button(page, "+1 hour"), 2.0)
    clip.caption("03:52: the low has begun. No repeat alert during the low or within the 60-minute cooldown")
    clip.hold(8)
    clip.click(check(page, "Simulate a sensor dropout"), 2)
    clip.caption("Sensor dropout: with the last 45 minutes missing, the twin says unavailable instead of guessing")
    clip.hold(8)
    clip.click(check(page, "Simulate a sensor dropout"), 1.5)
    clip.click(check(page, "Synthetic context update: GLP-1"), 2)
    clip.scroll(1000)
    clip.caption("A synthetic GLP-1 start fires the dose-review rule; it does not refit the physiology")
    clip.hold(9)
    clip.scroll(-1000)
    clip.click(check(page, "Synthetic context update: GLP-1"), 1.5)
    clip.click(button(page, "Reset to calibration end"), 2.5)
    clip.caption("Reset: the clock returns to the end of the 5-day calibration; nothing carries over")
    clip.hold(6)
    clip.caption("")


def tour_clip(clip):
    page = clip.page
    clip.click(page.get_by_role("tab", name="Clinic list (synthetic)", exact=True), 3)
    clip.caption("Clinic list (synthetic patients): review triggers and safety notes, judged as of the demo date")
    clip.hold(9)
    clip.click(page.get_by_role("tab", name="Patient (synthetic)", exact=True), 3)
    clip.caption("Meena (synthetic): medicines and dated records; each trigger shows its reason and label note")
    clip.hold(6)
    clip.scroll(500)
    clip.hold(5)
    clip.scroll(500)
    clip.hold(5)
    clip.scroll_top()
    clip.click(page.get_by_role("tab", name="Dose scenarios (illustrative)", exact=True), 4)
    clip.caption("Dose scenarios: options side by side on her core, under stated assumptions (illustrative)")
    clip.hold(7)
    clip.scroll(420)
    clip.hold(5)
    slider = page.locator("[data-testid='stSlider'], [data-testid='stSelectSlider']").filter(
        has_text=re.compile("Process-noise")).get_by_role("slider")
    if slider.count():
        clip.glide(slider.first)
        slider.first.focus()
        page.keyboard.press("ArrowRight")
        clip.ready(4)
        clip.caption("More process noise: the ranking holds, but the percentages move with the assumption")
        clip.hold(8)
    clip.scroll_top()
    clip.click(page.get_by_role("tab", name="Evidence (retrospective)", exact=True), 3)
    clip.caption("Evidence: retrospective results against baselines, including the negative findings")
    clip.hold(7)
    clip.scroll(600)
    clip.hold(5)
    clip.scroll_top()
    clip.click(page.get_by_role("tab", name="Data formats (example shapes)", exact=True), 3)
    clip.caption("Example record shapes: a FHIR-style bundle and smartwatch data (display only, not validated)")
    clip.hold(5)
    clip.scroll(450)
    clip.hold(5)
    clip.scroll(450)
    clip.hold(4)
    clip.caption("")


def caption_changes(path: Path, fps: int = 10) -> list:
    """Video times (s) at which the caption bar appears or its text changes.

    The recording's timeline runs a few percent slower than the wall clock of the script, so cue
    times are re-measured on the encoded video: the white text inside the navy caption bar is
    compared frame to frame, and a low overlap marks a new caption."""
    import numpy as np
    w, h, y0 = W // 2, 100, H // 2 - 100
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={fps},scale={W // 2}:{H // 2},crop={w}:{h}:0:{y0}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, i, times = None, 0, []
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.int16)
        navy = (abs(fr[..., 0] - 15) < 30) & (abs(fr[..., 1] - 39) < 30) & (abs(fr[..., 2] - 71) < 30)
        mask = None
        if navy.sum() > 5000:
            ys, xs = np.where(navy)
            mask = np.zeros(navy.shape, bool)
            r0, r1, c0, c1 = ys.min() + 2, ys.max() - 1, xs.min() + 2, xs.max() - 1
            mask[r0:r1, c0:c1] = fr[r0:r1, c0:c1].min(axis=2) > 170
        t = i / fps
        if mask is not None and prev is None:
            times.append(t)
        elif mask is not None and prev is not None:
            union = (mask | prev).sum()
            iou = (mask & prev).sum() / union if union else 1.0
            if 0.15 < iou < 0.5 and (not times or t - times[-1] > 2.0):
                times.append(t)
        prev, i = mask, i + 1
    proc.wait()
    return times


def align_cues(entry: dict) -> dict:
    """Add `t_video` to each cue: the detected caption change nearest its expected video time."""
    found = caption_changes(ROOT / entry["file"])
    offset = (found[0] - entry["cues"][0]["t"]) if found else 0.0
    for cue in entry["cues"]:
        pred = cue["t"] + offset
        near = [t for t in found if pred - 1.5 <= t <= pred + 3.0]
        cue["t_video"] = round(near[0], 1) if near else round(pred, 1)
        offset = cue["t_video"] - cue["t"]
    return entry


def record(name, fn):
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp),
                            record_video_size={"width": W, "height": H})
        ctx.add_init_script(OVERLAY)
        page = ctx.new_page()
        t_video = time.monotonic()  # the recording starts with the page
        page.goto(URL, wait_until="networkidle")
        clip = Clip(page, name)
        clip.ready(3)
        clip.t0 = time.monotonic()  # the clip is cut to start here, once the dashboard has loaded
        trim = clip.t0 - t_video
        fn(clip)
        clip.hold(1)
        video = page.video.path()
        ctx.close()
        b.close()
    out = DOCS / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim:.2f}", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-r", "25", "-an", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout)
    print(f"{out.name}: {dur:.1f} s, {len(clip.cues)} cues", flush=True)
    return {"file": f"docs/{out.name}", "duration_s": round(dur, 1), "cues": clip.cues}


if __name__ == "__main__":
    if "--align-only" in sys.argv:  # re-measure cue times on the existing clips
        cues = json.loads((DOCS / "demo_cues.json").read_text())
    else:
        old = json.loads((DOCS / "demo_cues.json").read_text()) if (DOCS / "demo_cues.json").exists() else {}
        only = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")), None)
        cues = {"live_replay": record("demo_1_live_replay", replay_clip) if only in (None, "replay") else old["live_replay"],
                "clinic_tour": record("demo_2_clinic_tour", tour_clip) if only in (None, "tour") else old["clinic_tour"]}
    cues = {k: align_cues(v) for k, v in cues.items()}
    (DOCS / "demo_cues.json").write_text(json.dumps(cues, indent=1))
    for k, v in cues.items():
        print(k, [(c["t"], c["t_video"]) for c in v["cues"]])
