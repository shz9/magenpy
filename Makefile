.PHONY: fix fix-python fix-cython fix-cpp lint lint-python lint-cython lint-cpp build build-inplace test test-inplace test-plink-docker dist redist install install-from-source clean uninstall publish-test publish

# Install requirements-lint.txt in an active virtual environment first.
fix: fix-python fix-cython fix-cpp

fix-python:
	ruff check --fix --exit-zero .
	ruff format .

fix-cython:
	python tools/format_cython.py

fix-cpp:
	git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hpp' '*.hxx' | \
		xargs -0 clang-format -i

lint: lint-python lint-cython lint-cpp

lint-python:
	ruff check .
	ruff format --check .

lint-cython:
	cython-lint --no-pycodestyle magenpy

# Lint tracked native sources; generated Cython .c/.cpp outputs stay excluded.
lint-cpp:
	git ls-files -z -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hpp' '*.hxx' | \
		xargs -0 clang-format --dry-run --Werror
	git ls-files -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hpp' '*.hxx' | \
		cppcheck --file-list=- --std=c++17 --enable=warning,performance,portability \
		--error-exitcode=1 --force --inline-suppr --suppress=missingIncludeSystem

build:
	python3 setup.py build

build-inplace:
	python3 setup.py build_ext --inplace

test-inplace:
	PYTHONPATH=. pytest

test:
	python -m pytest

test-plink-docker:
	bash tests/scripts/run_plink_backend_docker.sh

dist:
	python setup.py sdist bdist_wheel

redist: clean dist

install:
	python -m pip install .

install-from-source: dist
	python -m pip install dist/magenpy-*.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	$(RM) -r magenpy/stats/ld/*.c magenpy/stats/score/*.cpp
	$(RM) -r magenpy/stats/ld/*.so magenpy/stats/score/*.so
	$(RM) -r .pytest_cache .tox temp output
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	python -m pip uninstall magenpy

publish-test:
	python -m twine upload -r testpypi dist/* --verbose

publish:
	python -m twine upload dist/* --verbose
