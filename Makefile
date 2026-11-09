PYTHON ?= .venv/bin/python
JUPYTER ?= .venv/bin/jupyter

.PHONY: help install download notebook execute reports supplementary check validate rebuild-notebook

help:
	@echo "install           Install direct dependencies into .venv"
	@echo "download          Download pinned Kaggle inputs (credentials required)"
	@echo "notebook          Open the executed notebook in JupyterLab"
	@echo "execute           Re-run all analysis and regenerate reports"
	@echo "reports           Regenerate reports from committed results"
	@echo "supplementary     Gap trends, factor comparison, literature and weather figures (no raw data)"
	@echo "check             Check committed artifacts without raw data"
	@echo "validate          Independently verify results using raw data"
	@echo "rebuild-notebook  Overwrite notebook source from generator scripts"

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/download_expansion.py
	$(PYTHON) scripts/download_training_logs.py

notebook:
	$(JUPYTER) lab marathon_analysis.ipynb

execute:
	$(JUPYTER) nbconvert --to notebook --execute --inplace marathon_analysis.ipynb --ExecutePreprocessor.timeout=600

reports:
	$(PYTHON) scripts/render_berlin_report.py
	$(PYTHON) scripts/render_report.py

supplementary:
	$(PYTHON) scripts/supplementary_analysis.py

check:
	$(PYTHON) scripts/check_artifacts.py

validate:
	$(PYTHON) scripts/validate_project.py

rebuild-notebook:
	$(PYTHON) scripts/build_notebook.py
