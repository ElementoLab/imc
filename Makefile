.DEFAULT_GOAL := all

all: install test

clean_build:
	rm -rf build/

clean_dist:
	rm -rf dist/

clean_eggs:
	rm -rf *.egg-info

clean_mypy:
	rm -rf .mypy_cache/

clean_docs:
	rm -rf docs/build/*

clean_tests:
	rm -rf /tmp/pytest*

clean: clean_dist clean_eggs clean_build clean_mypy clean_docs

_install:
	# python setup.py sdist
	# python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	# python -m pip install dist/*-py3-none-any.whl --user --upgrade
	python -m pip install .

install:
	${MAKE} clean
	${MAKE} _install
	${MAKE} clean

docs:
	${MAKE} -C docs html
	xdg-open docs/build/html/index.html

test:
	python -m pytest imc/ -m "not slow"

sync:
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/imc

.PHONY : clean_build clean_dist clean_eggs clean_mypy clean_docs clean_tests \
clean _install install clean_docs docs test sync
