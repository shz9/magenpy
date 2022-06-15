.PHONY: build dist redist install install-from-source clean uninstall

build:
	python3 setup.py build

dist:
	python3 setup.py sdist bdist_wheel

redist: clean dist

install:
	pip install .

install-from-source: dist
	pip install dist/magenpy-*.tar.gz

clean:
	$(RM) -r build dist *.egg-info
	$(RM) -r magenpy/LDMatrix.c magenpy/utils/c_utils.c
	find . -name __pycache__ -exec rm -r {} +

uninstall:
	pip uninstall magenpy