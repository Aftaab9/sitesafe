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

FILE_LIST_PLACEHOLDER

## Numbers

Every number in the report, model card and slides comes from files in `results/` (for example `results/eval/compliance_base_test.json`, `results/synth/experiment_diffusion.csv`, `results/eval/fairness_audit.csv`, `results/vlm_baseline/vlm_baseline.json`). No metric was estimated. Figures are drawn from those files by `scripts/make_report_figures.py`; screenshots are taken from the live GitHub Pages app by `scripts/take_screenshots.py`.

## Things to know before submitting

- **Demo video:** not included. The project was presented in class, and you said a video upload is not needed.
- **Skin-tone audit:** the 230 test-split workers were sorted into lighter / medium / darker / not visible by eye from head-and-shoulder crops, because the automatic colour-based grouping failed. One person did the labelling and the darker group has only 12 people; the report says so. If you want to check the labels, they are in `data/splits/test_tags.csv` (column `skin_tone`).
- **Optional LFM3-Omni feature:** it only works on a computer that runs Ollama with `lfm3-omni` and allows the site. On this laptop only the Vercel mirror is allowed so far; to use it on the GitHub Pages URL, run `setx OLLAMA_ORIGINS "https://aftaab9.github.io,https://sitesafe-phi.vercel.app"` and restart Ollama. Visitors without Ollama do not see the option, and nothing else depends on it.
- **Platform:** Vercel is not on the brief's approved list, so the GitHub Pages URL is the one to give. GitHub Pages cannot send cross-origin isolation headers, so WebAssembly runs on one thread there; with WebGPU (most laptops) a photo still takes about 0.3 s.
- **Report word count:** 3,991 words in the body (sections 1 to 8), not counting figure captions and references.
