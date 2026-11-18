"""SiteSafe - Streamlit fallback app (Streamlit Community Cloud).

Same ONNX model and the same Python post-processing (sitesafe/) as the evaluation scripts.
Images and clips are processed in memory for this request only and are never written to disk
or logged. Run locally with:  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sitesafe.classes import FORKLIFT, FORKLIFT_MIN_CONF, PPE_ITEMS, SITE_PROFILES  # noqa: E402
from sitesafe.detector import Detector  # noqa: E402
from sitesafe.matching import MatchConfig, match_people, summarise  # noqa: E402
from sitesafe.privacy import FaceDetector, blur_regions, privacy_boxes  # noqa: E402
from sitesafe.tracking import Tracker, TrackerConfig  # noqa: E402
from sitesafe.zones import zone_events  # noqa: E402

MODEL = ROOT / "models" / "sitesafe_base.onnx"
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
COLOR = {"compliant": (47, 214, 123), "non_compliant": (255, 77, 79), "unsure": (91, 192, 255)}
LABEL = {"compliant": "OK", "non_compliant": "PPE MISSING", "unsure": "CHECK"}
PROFILE_NAMES = {"general": "General site (helmet, vest)", "full": "Full PPE (+ gloves, boots)",
                 "cutting_grinding": "Cutting / grinding (+ goggles)"}

st.set_page_config(page_title="SiteSafe: PPE compliance with privacy", page_icon="\U0001F9BA", layout="wide")


@st.cache_resource(show_spinner="Loading the on-server models (once)...")
def load_models():
    return Detector(str(MODEL), threads=2), FaceDetector(str(YUNET))


@st.cache_data
def thresholds() -> dict:
    import json
    f = ROOT / "results" / "eval" / "calibration.json"
    return {int(k): float(v) for k, v in json.loads(f.read_text())["class_thresholds"].items()} if f.exists() else {}


def analyse(rgb: np.ndarray, profile: str, conf: float, zone, fork_dist: float, strict: bool, blur: bool):
    det, faces = load_models()
    t0 = time.perf_counter()
    dets = det(rgb, conf)
    people = match_people(dets, SITE_PROFILES[profile], MatchConfig(class_thresholds=thresholds() or None))
    forklifts = [d.box for d in dets if d.cls == FORKLIFT and d.conf >= FORKLIFT_MIN_CONF]
    events = zone_events([p.box for p in people], forklifts, [zone] if zone else [], fork_dist)
    # Detection ran on the original pixels in memory; blurring is applied only to what is shown.
    shown = blur_regions(rgb, privacy_boxes(faces(rgb), [p.box for p in people], strict)) if blur else rgb.copy()
    return dets, people, events, shown, (time.perf_counter() - t0) * 1000


def draw(img: np.ndarray, people, events, zone, labels=None) -> np.ndarray:
    out = img.copy()
    s = max(1.0, max(out.shape[:2]) / 900)
    if zone:
        pts = np.array(zone, np.int32)
        hot = any(e["zones"] for e in events)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (255, 77, 79) if hot else (255, 122, 26))
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
        cv2.polylines(out, [pts], True, (255, 77, 79) if hot else (255, 122, 26), int(3 * s))
    for i, p in enumerate(people):
        x1, y1, x2, y2 = [int(v) for v in p.box]
        status = labels[i][1] if labels else p.status
        col = COLOR[status]
        if status == "compliant":
            cv2.rectangle(out, (x1, y1), (x2, y2), col, int(3 * s))
        else:  # dashed for missing, dotted for unsure: never colour alone
            step = int((18 if status == "non_compliant" else 8) * s)
            for x in range(x1, x2, step * 2):
                cv2.line(out, (x, y1), (min(x + step, x2), y1), col, int(3 * s)); cv2.line(out, (x, y2), (min(x + step, x2), y2), col, int(3 * s))
            for y in range(y1, y2, step * 2):
                cv2.line(out, (x1, y), (x1, min(y + step, y2)), col, int(3 * s)); cv2.line(out, (x2, y), (x2, min(y + step, y2)), col, int(3 * s))
        if events[i]["zones"] or events[i]["near_forklift"]:
            cv2.rectangle(out, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 122, 26), int(2 * s))
        name = labels[i][0] if labels else f"W{i + 1}"
        missing = labels[i][2] if labels else p.missing
        text = f"{name} {LABEL[status]}" + (f": no {', '.join(missing)}" if missing else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, max(1, int(1.5 * s)))
        ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 8
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), col, -1)
        cv2.putText(out, text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (11, 13, 16), max(1, int(1.5 * s)), cv2.LINE_AA)
    return out


def summary_table(s: dict, zone_n: int, fork_n: int) -> pd.DataFrame:
    rows = [("Workers", s["persons"]), ("Compliant", s["compliant"]), ("PPE missing", s["non_compliant"]),
            ("Check manually", s["unsure"]), ("Compliance %", "n/a" if s["compliance_pct"] is None else s["compliance_pct"])]
    rows += [(f"Missing {i}", s["missing_by_item"][i]) for i in PPE_ITEMS]
    rows += [("Forklifts", s["forklifts"]), ("Workers in danger zone", zone_n), ("Workers near forklift", fork_n)]
    return pd.DataFrame(rows, columns=["metric", "value"])


def metrics_row(s: dict):
    c = st.columns(4)
    c[0].metric("Compliance", "n/a" if s["compliance_pct"] is None else f"{s['compliance_pct']} %")
    c[1].metric("Compliant", s["compliant"])
    c[2].metric("PPE missing", s["non_compliant"])
    c[3].metric("Check manually", s["unsure"])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("\U0001F9BA SiteSafe")
st.sidebar.caption("PPE compliance with privacy")
profile = st.sidebar.selectbox("Site profile (which PPE is mandatory)", list(PROFILE_NAMES), format_func=PROFILE_NAMES.get)
conf = st.sidebar.slider("Detection sensitivity (min. confidence)", 0.10, 0.60, 0.25, 0.05)
blur = st.sidebar.toggle("Blur faces", True)
strict = st.sidebar.toggle("Strict privacy (blur head area when no face is found)", True)
fork_dist = st.sidebar.slider("Forklift safety distance (x person height)", 0.1, 2.0, 0.5, 0.1)
with st.sidebar.expander("Danger zone (rectangle, % of the image)"):
    use_zone = st.checkbox("Enable a danger zone")
    zx = st.slider("Left to right", 0, 100, (55, 100))
    zy = st.slider("Top to bottom", 0, 100, (40, 100))
st.sidebar.info("Photos and clips are processed in memory and discarded. Nothing is stored. "
                "For on-device processing, live camera and polygon zones, use the main web app.")

st.title("SiteSafe: who is missing PPE?")
st.caption("Solid green box: compliant. Dashed red: required PPE missing. Dotted blue: the model is not sure, please check. "
           "This tool supports the safety officer; it must not be used to monitor or discipline individuals.")

tab_photo, tab_clip, tab_help = st.tabs(["Photo", "Clip (10-30 s)", "Help"])

with tab_photo:
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png", "webp"])
    cam = c2.camera_input("...or take one")
    src = up or cam
    if src:
        rgb = np.array(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
        if max(rgb.shape[:2]) > 1600:
            r = 1600 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * r), round(rgb.shape[0] * r)), interpolation=cv2.INTER_AREA)
        H, W = rgb.shape[:2]
        zone = [(zx[0] * W / 100, zy[0] * H / 100), (zx[1] * W / 100, zy[0] * H / 100),
                (zx[1] * W / 100, zy[1] * H / 100), (zx[0] * W / 100, zy[1] * H / 100)] if use_zone else None
        dets, people, events, shown, ms = analyse(rgb, profile, conf, zone, fork_dist, strict, blur)
        s = summarise(people, dets)
        zone_n = sum(bool(e["zones"]) for e in events); fork_n = sum(e["near_forklift"] for e in events)
        left, right = st.columns([3, 2])
        annotated = draw(shown, people, events, zone)
        left.image(annotated, use_container_width=True)
        left.caption(f"Answered in {ms:.0f} ms on the free CPU. Faces {'blurred' if blur else 'NOT blurred'}.")
        with right:
            metrics_row(s)
            if s["persons"] and not s["non_compliant"] and not s["unsure"]:
                st.success("Site clear: everyone in view is wearing the required PPE.")
            if s["unsure"]:
                st.info(f"{s['unsure']} worker(s) could not be decided with confidence. That is expected for people who are "
                        "far away, partly hidden or in poor light. Please take a look yourself.")
            if zone_n or fork_n:
                st.warning(f"{zone_n} worker(s) inside the danger zone, {fork_n} close to a forklift.")
            st.dataframe(pd.DataFrame([{"worker": f"W{i + 1}", "status": LABEL[p.status], "missing": ", ".join(p.missing),
                                        "not sure about": ", ".join(p.unsure_items), "confidence": round(p.conf, 2)}
                                       for i, p in enumerate(people)]), hide_index=True, use_container_width=True)
            table = summary_table(s, zone_n, fork_n)
            st.download_button("Download incident summary (CSV)", table.to_csv(index=False), "sitesafe-incident.csv", "text/csv")
            buf = io.BytesIO(); Image.fromarray(annotated).save(buf, "JPEG", quality=88)
            st.download_button("Download blurred image", buf.getvalue(), "sitesafe-blurred.jpg", "image/jpeg")

with tab_clip:
    st.write("Frames are sampled a few times per second. Each worker keeps an anonymous number, and a violation is "
             "only raised when it persists in 3 of the last 5 sampled frames.")
    vid = st.file_uploader("Upload a clip (mp4, mov, webm)", type=["mp4", "mov", "webm", "avi", "m4v"])
    fps_s = st.select_slider("Frames analysed per second", [2, 3, 5], 3)
    if vid and st.button("Analyse clip", type="primary"):
        # OpenCV needs a file path: the temp file is deleted as soon as the clip is read.
        with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as tf:
            tf.write(vid.getvalue()); tmp = tf.name
        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / fps_s))
        limit = int(min(n, fps * 60))
        tracker = Tracker(TrackerConfig(window=5, min_hits=3))
        required = SITE_PROFILES[profile]
        workers, timeline, preview = {}, [], None
        bar = st.progress(0.0, "Analysing...")
        holder = st.empty()
        t_start = time.perf_counter()
        for idx in range(0, limit, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if max(rgb.shape[:2]) > 960:
                r = 960 / max(rgb.shape[:2]); rgb = cv2.resize(rgb, (round(rgb.shape[1] * r), round(rgb.shape[0] * r)))
            H, W = rgb.shape[:2]
            zone = [(zx[0] * W / 100, zy[0] * H / 100), (zx[1] * W / 100, zy[0] * H / 100),
                    (zx[1] * W / 100, zy[1] * H / 100), (zx[0] * W / 100, zy[1] * H / 100)] if use_zone else None
            dets, people, events, shown, _ = analyse(rgb, profile, conf, zone, fork_dist, strict, blur)
            seen = tracker.update(people)
            labels, ev, ppl, bad = [], [], [], 0
            for t in seen:
                status = tracker.smoothed_status(t, required); missing = tracker.smoothed_missing(t, required)
                e = events[people.index(t.last)]
                w = workers.setdefault(t.id, {"frames": 0, "ok": 0, "missing": set(), "zone": False, "forklift": False, "violation": False})
                w["frames"] += 1; w["ok"] += status == "compliant"; w["missing"].update(missing)
                w["violation"] |= status == "non_compliant"; w["zone"] |= bool(e["zones"]); w["forklift"] |= e["near_forklift"]
                bad += status == "non_compliant"
                labels.append((f"W{t.id}", status, missing)); ev.append(e); ppl.append(t.last)
            timeline.append({"t (s)": round(idx / fps, 2), "workers in view": len(seen), "PPE missing": bad})
            preview = draw(shown, ppl, ev, zone, labels)
            if len(timeline) % 3 == 0:
                holder.image(preview, use_container_width=True)
            bar.progress(min(1.0, (idx + step) / limit), f"Analysing... {idx / fps:.1f} s")
        cap.release(); Path(tmp).unlink(missing_ok=True)
        total = time.perf_counter() - t_start
        bar.empty()
        if preview is not None:
            holder.image(preview, caption="Last analysed frame", use_container_width=True)
        stable = {k: v for k, v in workers.items() if v["frames"] >= min(3, len(timeline))}
        status_of = lambda w: "non_compliant" if w["violation"] else "compliant" if w["ok"] >= w["frames"] / 2 else "unsure"  # noqa: E731
        st_list = [status_of(w) for w in stable.values()]
        comp, non, uns = st_list.count("compliant"), st_list.count("non_compliant"), st_list.count("unsure")
        s = {"persons": len(stable), "compliant": comp, "non_compliant": non, "unsure": uns,
             "compliance_pct": round(100 * comp / (comp + non), 1) if comp + non else None,
             "missing_by_item": {i: sum(i in w["missing"] for w in stable.values()) for i in PPE_ITEMS}, "forklifts": 0}
        metrics_row(s)
        st.caption(f"{len(timeline)} frames in {total:.1f} s ({1000 * total / max(1, len(timeline)):.0f} ms per frame) on the free CPU.")
        tl = pd.DataFrame(timeline)
        st.subheader("Violations over time")
        st.area_chart(tl.set_index("t (s)"))
        st.subheader("Anonymous worker log")
        log = pd.DataFrame([{"worker": f"W{k}", "status": LABEL[status_of(w)], "missing": ", ".join(sorted(w["missing"])),
                             "danger zone": w["zone"], "near forklift": w["forklift"], "frames seen": w["frames"]} for k, w in stable.items()])
        st.dataframe(log, hide_index=True, use_container_width=True)
        zone_n = sum(w["zone"] for w in stable.values()); fork_n = sum(w["forklift"] for w in stable.values())
        st.download_button("Download incident summary (CSV)",
                           summary_table(s, zone_n, fork_n).to_csv(index=False) + "\n" + tl.to_csv(index=False), "sitesafe-clip.csv", "text/csv")

with tab_help:
    st.markdown("""
**What it does.** SiteSafe finds each worker in a photo or clip and checks whether the PPE required by the
site profile is visible on them. It reports a compliance percentage and counts by PPE type.

**Reading the result.** Green solid box: compliant. Red dashed box: a required item is visibly missing.
Blue dotted box: the model is not confident, or a required item is out of view. "Check" is not an error;
it is the model handing the decision to you.

**Privacy.** Faces are blurred before anything is shown. The upload is processed in memory for this request and
discarded. No image, clip or result is stored or logged. Worker numbers are anonymous and reset every time.

**Limits.** Small or distant workers, crowds, night and bad weather reduce accuracy. Forklift distance is
estimated in the image without depth. A missed "no helmet" is the costly error, so the model prefers "check"
over a wrong green.

**Do not** use this tool to identify, monitor or discipline individual workers. Tell workers what the check is for.

Model: YOLO11n fine-tuned on Construction-PPE (Ultralytics, CC BY 4.0). Face detector: YuNet (MIT). Code: AGPL-3.0.
""")
