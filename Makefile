
clean_dist:
	-rm -rf dist/

clean_eggs:
	-rm -r *.egg-info

clean: clean_dist clean_eggs

_install:
	python setup.py sdist
	python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	python -m pip install dist/*-py3-none-any.whl --user --upgrade


install: clean _install clean

run:
	python imcpipeline/runner.py \
		--divvy slurm \
		metadata/annotation.csv \
			--ilastik-model models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/panel_markers.csv \
			--cellprofiler-exec "source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"

run_locally:
	python imcpipeline/runner.py \
		--divvy local \
		metadata/annotation.csv \
			--ilastik-model models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/panel_data.csv \
			--container docker

checkfailure:
	grep -H "Killed" submission/*.log && \
	grep -H "Error" submission/*.log && \
	grep -H "CANCELLED" submission/*.log && \
	grep -H "exceeded" submission/*.log

fail: checkfailure

checksuccess:
	ls -hl processed/*/cpout/cell.csv

succ: checksuccess
