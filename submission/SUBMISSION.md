# SiteSafe: submission (MAIB AI 221, Project 6)

## Links

| | |
|---|---|
| Live app (approved platform: GitHub Pages) | https://aftaab9.github.io/sitesafe/ |
| Mirror (Vercel, not on the approved list, kept only as a mirror) | https://sitesafe-phi.vercel.app |
| Fallback | Streamlit app in the repository, `streamlit run streamlit_app/app.py` (runs locally; not deployed) |
| GitHub repository | https://github.com/Aftaab9/sitesafe |

The app is a static site: all models run in the visitor's browser, so there is no server to sleep or cold-start. Images never leave the device.

## What is in the zip

Top level:

- `Technical_Report.docx`: technical report, 3,991 words in the body, 12 figures (training curves, confusion matrix, baselines, calibration, with vs without synthetic data, subgroups, app screenshots)
- `Model_Card.docx`: one-page model card
- `SiteSafe_Presentation.pptx`: 14 slides for a 10-minute talk, speaker notes with timings on every slide
- `SUBMISSION.md`: this file

`screenshots/` (live app at 1920 x 1080, taken with Playwright):

- `screenshots/01_landing.png`
- `screenshots/02_photo_compliant_crew.png`
- `screenshots/02b_photo_compliant_crew_full.png`
- `screenshots/03_photo_missing_helmets.png`
- `screenshots/04_face_blur_before_after.png`
- `screenshots/05_danger_zone.png`
- `screenshots/06_not_sure_state.png`
- `screenshots/07_failure_case.png`
- `screenshots/08_explain_heatmap.png`
- `screenshots/09_pdf_report.png`
- `screenshots/10_clip_timeline.png`
- `screenshots/10b_clip_summary_full.png`
- `screenshots/11_dashboard.png`
- `screenshots/11b_dashboard_full.png`
- `screenshots/12_weather_lab.png`
- `screenshots/13_inspector_challenge.png`
- `screenshots/14_model_card.png`
- `screenshots/14b_model_card_full.png`
- `screenshots/15_help.png`
- `screenshots/16_live_camera_start.png`
- `screenshots/incident_example.csv`
- `screenshots/incident_report_example.pdf`

`figures/` (drawn from `results/` by `scripts/make_report_figures.py`):

- `figures/fig_baselines.png`
- `figures/fig_class_counts.png`
- `figures/fig_confusion_matrix.png`
- `figures/fig_reliability.png`
- `figures/fig_subgroups.png`
- `figures/fig_synthetic_with_without.png`
- `figures/fig_training_curves.png`

`sitesafe-repo/`: clean copy of the GitHub repository (tracked files only: no datasets, virtual environment, node_modules or training runs). Files per folder:

- `(root files)`: 9
- `.github`: 2
- `colab`: 8
- `data`: 41
- `docs`: 8
- `models`: 3
- `notebooks`: 3
- `results`: 171
- `scripts`: 36
- `sitesafe`: 11
- `streamlit_app`: 1
- `tests`: 4
- `web`: 174

Left out of the repo copy to keep the zip small (all still in the GitHub repository): the YOLO11s comparison model (38 MB), the two ablation models of the synthetic-data experiment, duplicate copies of the synth and plus models (the app's copies in `web/public/models/` are included), the Colab code bundles, duplicate .docx files and Ultralytics' sample-batch images. The datasets are downloaded by `scripts/download_data.py`.

## Numbers

Every number in the report, model card and slides comes from files in `results/` (for example `results/eval/compliance_base_test.json`, `results/synth/experiment_diffusion.csv`, `results/eval/fairness_audit.csv`, `results/vlm_baseline/vlm_baseline.json`). No metric was estimated. Figures are drawn from those files by `scripts/make_report_figures.py`; screenshots are taken from the live GitHub Pages app by `scripts/take_screenshots.py`.

## Things to know before submitting

- **Demo video:** not included. The project was presented in class, and you said a video upload is not needed.
- **Skin-tone audit:** the 230 test-split workers were sorted into lighter / medium / darker / not visible by eye from head-and-shoulder crops, because the automatic colour-based grouping failed. One person did the labelling and the darker group has only 12 people; the report says so. If you want to check the labels, they are in `data/splits/test_tags.csv` (column `skin_tone`).
- **Optional LFM3-Omni feature:** it only works on a computer that runs Ollama with `lfm3-omni` and allows the site. On this laptop only the Vercel mirror is allowed so far; to use it on the GitHub Pages URL, run `setx OLLAMA_ORIGINS "https://aftaab9.github.io,https://sitesafe-phi.vercel.app"` and restart Ollama. Visitors without Ollama do not see the option, and nothing else depends on it.
- **Platform:** Vercel is not on the brief's approved list, so the GitHub Pages URL is the one to give. GitHub Pages cannot send cross-origin isolation headers, so WebAssembly runs on one thread there; with WebGPU (most laptops) a photo still takes about 0.3 s.
- **Report word count:** 3,991 words in the body (sections 1 to 8), not counting figure captions and references.
