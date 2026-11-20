SEJM_TERM ?= 9
WORKERS   ?= 8
MAX_DAYS  ?=
TEXT      ?= Musimy obniżyć podatki dla przedsiębiorców i ograniczyć biurokrację.
TERMS     ?= 9,10

.DEFAULT_GOAL := help
.PHONY: help setup fetch fetch10 prepare-term prepare-combined train predict predict-club notebook clean clean-all all

help:
	@echo "Kolejnosc (druga kadencja = term 10):"
	@echo "  1) make setup"
	@echo "  2) make fetch10 WORKERS=16          # pobierz term 10 (dlugo trwa)"
	@echo "  3) make prepare-term SEJM_TERM=10   # merge+clean+label dla term 10"
	@echo "  4) make prepare-term SEJM_TERM=9    # odswiez term 9 jesli trzeba"
	@echo "  5) make prepare-combined           # polacz 9+10, zrob splity"
	@echo "  6) make train                      # TF-IDF side + club"
	@echo "  7) make notebook                   # presentation.ipynb"
	@echo ""
	@echo "Predykcja:"
	@echo "  make predict TEXT=\"...\""
	@echo "  make predict-club TEXT=\"...\""

setup:
	uv sync

fetch:
	uv run fetch-representatives --term $(SEJM_TERM)
	uv run fetch-speeches --term $(SEJM_TERM) --workers $(WORKERS) $(if $(MAX_DAYS),--max-days $(MAX_DAYS),)

fetch10:
	uv run fetch-representatives --term 10
	uv run fetch-speeches --term 10 --workers $(WORKERS) $(if $(MAX_DAYS),--max-days $(MAX_DAYS),)

prepare-term:
	uv run merge-speeches --term $(SEJM_TERM)
	uv run clean-speeches --input data/term$(SEJM_TERM)/dataset.csv --output data/term$(SEJM_TERM)/dataset_clean.csv
	uv run label-sides --input data/term$(SEJM_TERM)/dataset_clean.csv --output data/term$(SEJM_TERM)/dataset_labeled.csv

prepare-combined:
	uv run combine-datasets --terms $(TERMS) --output data/combined/dataset_labeled.csv
	uv run split-dataset --input data/combined/dataset_labeled.csv --output-dir data/combined/splits

train:
	uv run train-baseline --splits-dir data/combined/splits --output-dir models

predict:
	uv run predict-side "$(TEXT)"

predict-club:
	uv run predict-club "$(TEXT)"

notebook:
	uv run jupyter notebook src/data_mining_project/notebooks/presentation.ipynb

clean:
	rm -rf models data/term$(SEJM_TERM)/splits data/combined
	rm -f data/term$(SEJM_TERM)/dataset.csv data/term$(SEJM_TERM)/dataset_clean.csv data/term$(SEJM_TERM)/dataset_labeled.csv

clean-all:
	rm -rf data models

all: setup fetch10 prepare-term
	$(MAKE) prepare-term SEJM_TERM=9
	$(MAKE) prepare-combined
	$(MAKE) train
