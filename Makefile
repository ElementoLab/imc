
install:
	python setup.py sdist
	python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	python -m pip install dist/*-py3-none-any.whl --user --upgrade

clean:
	rm -rf dist/
	rm -r *.egg-info


run:
	python imcpipeline/runner.py \
		--divvy slurm \
		metadata/annotation.csv \
			--ilastik-model models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/DLBCL.csv

run_locally:
	python imcpipeline/runner.py \
		--divvy local \
		metadata/annotation.csv \
			--ilastik-model models/lymphoma/lymphoma.ilp \
			--csv-pannel metadata/DLBCL.csv \
			--container docker
