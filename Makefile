.DEFAULT_GOAL := all

all: install clean test

move_models_out:
	mv _models ../

move_models_in:
	mv ../_models ./

clean_build:
	rm -rf build/

clean_dist:
	rm -rf dist/

clean_eggs:
	rm -rf *.egg-info

clean: clean_dist clean_eggs clean_build

_install:
	python setup.py sdist
	python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	python -m pip install dist/*-py3-none-any.whl --user --upgrade

install:
	${MAKE} move_models_out
	${MAKE} clean
	${MAKE} _install
	${MAKE} clean
	${MAKE} move_models_in

test:
	python -m pytest imc/

run:
	python imcpipeline/runner.py \
		--divvy slurm \
		metadata/annotation.csv \
			--ilastik-model _models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/panel_markers.csv \
			--cellprofiler-exec \
				"source ~/.miniconda2/bin/activate && conda activate cellprofiler && cellprofiler"

run_locally:
	python imcpipeline/runner.py \
		--divvy local \
		metadata/annotation.csv \
			--ilastik-model _models/lymphoma/lymphoma.ilp \
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


.PHONY : move_models_out move_models_in clean_build clean_dist clean_eggs \
clean _install install run run_locally checkfailure fail checksuccess succ
