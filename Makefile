.PHONY: quick full diagnostics summarize validate figures check paper

quick:
	python experiments/multi_active_replay.py --suite quick --out experiments/quick_results

full:
	python experiments/multi_active_replay.py --suite full --out experiments/results
	python experiments/frontier_diagnostics.py --mode full --out experiments/results

diagnostics:
	python experiments/frontier_diagnostics.py --mode full --out experiments/results

summarize:
	python experiments/multi_active_replay.py --suite summarize --out experiments/results
	python experiments/frontier_diagnostics.py --mode summarize --out experiments/results

validate: summarize
	python experiments/validate_results.py

figures:
	python experiments/plot_results.py --results experiments/results --figures paper/figures

check: validate figures
	python -m py_compile experiments/multi_active_replay.py experiments/frontier_diagnostics.py experiments/plot_results.py experiments/validate_results.py

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
