from setuptools import setup, Extension, find_packages
import numpy as np
import os

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# ------------------------------------------------------
# Cython dependencies:


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("magenpy.stats.ld.c_utils",
              sources=["magenpy/stats/ld/c_utils.pyx"],
              extra_compile_args=["-ffast-math"],
              include_dirs=[np.get_include()],
              ),
    Extension("magenpy.LDMatrix",
              sources=["magenpy/LDMatrix.pyx"],
              extra_compile_args=["-ffast-math"],
              include_dirs=[np.get_include()],
              )
]
if cythonize is not None:
    compiler_directives = {
        "language_level": 3,
        "embedsignature": True,
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'cdivision': True
    }
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

# ------------------------------------------------------
# Read description/dependencies from file:

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-optional.txt") as fp:
    opt_requires = fp.read().strip().split("\n")

# ------------------------------------------------------

setup(
    name="magenpy",
    version="0.0.2",
    author="Shadi Zabad",
    author_email="shadi.zabad@mail.mcgill.ca",
    description="Modeling and Analysis of Statistical Genetics data in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shz9/magenpy",
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_dir={'': '.'},
    packages=find_packages(),
    package_data={'magenpy': ['data/*.bed', 'data/*.bim', 'data/*.fam', 'config/*.ini']},
    scripts=['bin/magenpy_ld', 'bin/magenpy_simulate'],
    install_requires=install_requires,
    extras_require={'full': opt_requires},
    ext_modules=extensions,
    zip_safe=False
)
