"""Capture the submission screenshots of the live app with Playwright (desktop, 1920 x 1080).

    python scripts/take_screenshots.py [--url https://aftaab9.github.io/sitesafe/]

Uses only the app's own sample gallery. Writes submission/screenshots/*.png.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "screenshots"


def wait_result(page, timeout=90):
    t = time.time()
    while time.time() - t < timeout:
        if page.locator("span.font-mono", has_text=" ms ·").count():
            return
        time.sleep(0.5)
    raise TimeoutError("no result")


def open_sample(page, url, sid):
    page.goto(f"{url}?s={sid}#/analyze?sample={sid}")  # a new query string forces a fresh page load
    wait_result(page)
    time.sleep(1.8)  # let the lock-on animation finish


def shot(page, name, full=False, el=None):
    path = OUT / f"{name}.png"
    page.mouse.move(1919, 1079)  # keep the pointer away from hover effects
    time.sleep(0.4)
    (el.screenshot(path=str(path)) if el is not None else page.screenshot(path=str(path), full_page=full))
    print("saved", path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://aftaab9.github.io/sitesafe/")
    ap.add_argument("--from-clip", action="store_true", help="skip the photo screenshots (they still run once to fill the dashboard)")
    a = ap.parse_args()
    url = a.url
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # A visible (off-screen) window gets the GPU; headless Chrome falls back to slow software rendering.
        b = p.chromium.launch(headless=False, args=["--enable-unsafe-webgpu", "--hide-scrollbars", "--window-position=-2400,0"])
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, accept_downloads=True)
        page = ctx.new_page()
        page.add_init_script("try{localStorage.setItem('sitesafe-settings',JSON.stringify({state:{profile:'general',conf:0.25,strictBlur:true,blurFaces:true,sound:false,forkliftDistance:0.5,theme:'dark',reducedMotion:true,detector:'base'},version:0}))}catch(e){}")

        page.goto(url); time.sleep(3)
        shot(page, "01_landing")

        open_sample(page, url, "r01")
        shot(page, "02_photo_compliant_crew")
        stage = page.locator(".hud-corners").first
        shot(page, "02b_photo_compliant_crew_full", full=True)

        open_sample(page, url, "r02")
        shot(page, "03_photo_missing_helmets")
        page.get_by_role("button", name="Privacy slider").click(); time.sleep(1)
        box = stage.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        shot(page, "04_face_blur_before_after", el=stage)
        page.get_by_role("button", name="Privacy slider").click(); time.sleep(0.5)

        # danger zone around the left part of the group
        page.get_by_role("button", name="Draw danger zone").click(); time.sleep(0.5)
        box = stage.bounding_box()
        for fx, fy in ((0.14, 0.55), (0.45, 0.55), (0.45, 0.98), (0.14, 0.98)):
            page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy); time.sleep(0.2)
        page.keyboard.press("Enter"); time.sleep(1.2)
        shot(page, "05_danger_zone")

        open_sample(page, url, "r11")
        shot(page, "06_not_sure_state")

        open_sample(page, url, "r13")
        shot(page, "07_failure_case")

        open_sample(page, url, "r01")
        page.get_by_role("button", name="Explain").click(); time.sleep(1.2)
        shot(page, "08_explain_heatmap", el=stage)
        page.get_by_role("button", name="Explain").click()
        with page.expect_download() as d:
            page.get_by_role("button", name="PDF report").click()
        pdf = OUT / "incident_report_example.pdf"
        d.value.save_as(str(pdf))
        with page.expect_download() as d:
            page.get_by_role("button", name="CSV").click()
        d.value.save_as(str(OUT / "incident_example.csv"))
        try:
            import fitz
            doc = fitz.open(str(pdf))
            doc[0].get_pixmap(dpi=110).save(str(OUT / "09_pdf_report.png")); print("saved 09_pdf_report.png")
        except Exception as e:  # noqa: BLE001
            print("pdf render skipped", e)

        # From here on, move between pages inside the app (no reload), so the session log feeds the dashboard.
        def go(route, wait):
            page.evaluate(f"location.hash = '#/{route}'"); time.sleep(wait)
        go("video", 3)
        page.get_by_role("button", name="Several violations").click(); time.sleep(5)
        page.get_by_role("button", name="Analyse clip").click()
        t = time.time()
        while time.time() - t < 600 and not page.locator("text=ms/frame").count():
            time.sleep(2)
        print("clip took", round(time.time() - t), "s")
        time.sleep(2)
        shot(page, "10_clip_timeline")
        shot(page, "10b_clip_summary_full", full=True)

        go("dashboard", 3)
        shot(page, "11_dashboard")
        shot(page, "11b_dashboard_full", full=True)

        go("weather", 3)
        t = time.time()
        while time.time() - t < 240 and page.locator("text=/\d+% \(clear/").count() < 2:
            time.sleep(2)
        time.sleep(3)
        shot(page, "12_weather_lab")

        go("challenge", 3)
        page.get_by_role("button", name="Start the challenge").click()
        t = time.time()
        while time.time() - t < 120 and not page.get_by_role("button", name="Lock in").count():
            time.sleep(1)
        time.sleep(2)
        shot(page, "13_inspector_challenge")

        go("model-card", 4)
        shot(page, "14_model_card")
        shot(page, "14b_model_card_full", full=True)
        go("help", 2)
        shot(page, "15_help")
        go("live", 3)
        shot(page, "16_live_camera_start")
        b.close()


if __name__ == "__main__":
    main()
