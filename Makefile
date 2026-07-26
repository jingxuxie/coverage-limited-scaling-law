.PHONY: quick full summarize validate figures check

quick:
	python experiments/multi_active_replay.py --suite quick --out experiments/quick_results

full:
	python experiments/multi_active_replay.py --suite full --out experiments/results

summarize:
	python experiments/multi_active_replay.py --suite summarize --out experiments/results

validate: summarize
	python experiments/validate_results.py

figures:
	python experiments/plot_results.py --results experiments/results --figures paper/figures

check: validate figures
	python -m py_compile experiments/multi_active_replay.py experiments/plot_results.py experiments/validate_results.py
