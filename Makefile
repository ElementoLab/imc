
clean:
	rm -rf dist/
	rm -r *.egg-info

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
			--cellprofiler-exec "source ~/.miniconda2/bin/activate; conda activate cellprofiler; cellprofiler"

run_locally:
	python imcpipeline/runner.py \
		--divvy local \
		metadata/annotation.csv \
			--ilastik-model models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/panel_data.csv \
			--container docker
