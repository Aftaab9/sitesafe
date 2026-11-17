# SiteSafe: one command per step. On Windows without make, use:  python tasks.py <target>
PY ?= python

.PHONY: setup data split stats baselines train export calibrate evaluate quantise synth synth-train studies metrics assets web test deploy-check all

setup:            ## install training requirements and web dependencies
	$(PY) -m pip install -r requirements-train.txt
	cd web && npm ci

data:             ## download Construction-PPE (CC BY 4.0) and YuNet, fix data.yaml
	$(PY) scripts/download_data.py

split:            ## seeded, stratified train / val / test split
	$(PY) scripts/make_split.py

stats:            ## dataset statistics and sample grid
	$(PY) scripts/dataset_stats.py

baselines:        ## pre-trained + colour rules, and zero-shot YOLO-World
	$(PY) scripts/baseline_rules.py --split test
	$(PY) scripts/baseline_zeroshot.py --split test

train:            ## fine-tune YOLO11n on real data (EPOCHS=100 on a GPU)
	$(PY) scripts/train.py --name base --epochs $(or $(EPOCHS),100)

export:           ## best checkpoint -> ONNX (opset 17)
	$(PY) scripts/export_onnx.py --weights runs/base/weights/best.pt --name sitesafe_base

calibrate:        ## temperature, reliability diagram, per-class "unsure" thresholds
	$(PY) scripts/calibrate.py

evaluate:         ## detection metrics and person-level compliance on the test split
	$(PY) scripts/evaluate.py --weights models/sitesafe_base.pt --split test --tag base
	$(PY) scripts/compliance_eval.py --tag base --split test

quantise:         ## int8 model, accuracy cost measured
	$(PY) scripts/quantize.py --name sitesafe_base

synth:            ## diffusion weather / night images + quality filter
	$(PY) scripts/synth_generate.py --n $(or $(N),200) --mode diffusion
	$(PY) scripts/synth_filter.py --mode diffusion

synth-train:      ## train real + synthetic with the same settings, then compare
	$(PY) scripts/train.py --name synth --data ppe_synth_diffusion.yaml --epochs $(or $(EPOCHS),100)
	$(PY) scripts/export_onnx.py --weights runs/synth/weights/best.pt --name sitesafe_synth
	$(PY) scripts/synth_experiment.py

studies:          ## fairness audit, failure gallery, smoothing, face blur, VLM hallucination
	$(PY) scripts/fairness_audit.py
	$(PY) scripts/failure_gallery.py
	$(PY) scripts/smoothing_study.py
	$(PY) scripts/blur_study.py
	$(PY) scripts/vlm_hallucination.py --n 60

metrics:          ## collect all results for the model card and report
	$(PY) scripts/build_metrics.py

assets:           ## package models, gallery, heat-maps, weather lab and clips for the web app
	$(PY) scripts/build_web_assets.py

web:              ## production build of the static app (web/dist)
	cd web && npm run build

test:             ## Python and TypeScript unit tests, type-check
	$(PY) -m pytest -q
	cd web && npm run typecheck && npm test

deploy-check:     ## what gets deployed: Vercel / Cloudflare Pages read web/, Streamlit reads streamlit_app/app.py
	cd web && npm run build && npx vite preview

all: data split stats baselines train export calibrate evaluate quantise synth synth-train studies metrics assets web
