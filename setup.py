from setuptools import setup, Extension, find_packages
import numpy as np
import os

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# ------------------------------------------------------
# Cython dependencies:

def no_cythonize(extensions, **_ignore):
    """
    Copied from:
    https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
    """
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
              include_dirs=[np.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
    # Not ready yet:
    # Extension("magenpy.stats.score.score_cpp",
    #          sources=["magenpy/stats/score/score_cpp.pyx"],
    #          include_dirs=[np.get_include()],
    #          language='c++'
    #          )
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

with open("requirements-test.txt") as fp:
    test_requires = fp.read().strip().split("\n")

with open("requirements-docs.txt") as fp:
    docs_requires = fp.read().strip().split("\n")

# ------------------------------------------------------

setup(
    name="magenpy",
    version="0.1.0",
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    package_dir={'': '.'},
    packages=find_packages(),
    python_requires=">=3.8",
    package_data={'magenpy': ['data/*.bed', 'data/*.bim', 'data/*.fam',
                              'data/ukb_height_chr22.fastGWA.gz',
                              'config/*.ini']},
    scripts=['bin/magenpy_ld', 'bin/magenpy_simulate'],
    install_requires=install_requires,
    extras_require={'opt': opt_requires, 'test': test_requires, 'docs': docs_requires},
    ext_modules=extensions,
    zip_safe=False
)
