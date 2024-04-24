The `magenpy` software is written in `Cython/Python3`.
The software is designed to be used in a variety of computing environments, including local workstations, 
shared computing environments, and cloud-based computing environments. Because of the dependencies on `Cython`, you need 
to ensure that a `C/C++` Compiler (with appropriate flags) is present on your system.

## Requirements

Building the `magenpy` package requires the following dependencies:

* `Python` (>=3.8)
* `C/C++` Compiler
* `Cython` (>=0.29.21)
* `NumPy` (>=1.19.5)

### Setting up the environment with `conda`

If you can use `Anaconda` or `miniconda` to manage your Python environment, we recommend using them to create 
a new environment with the required dependencies as follows:

```bash
python_version=3.11  # Change python version here if needed
conda create --name "magenpy_env" -c anaconda -c conda-forge python=$python_version compilers openblas -y
conda activate magenpy_env
```

Using `conda` to set up and manage your environment is especially *recommended* if you have trouble compiling 
the `C/C++` extensions on your system.

## Installation

### Using `pip`

The package is available for easy installation via the Python Package Index (`pypi`) can be installed using `pip`:

```bash
python -m pip install magenpy>=0.1
```

### Building from source

You may also build the package from source, by cloning the repository and running the `make install` command:

```bash
git clone https://github.com/shz9/magenpy.git
cd magenpy
make install
```

### Using a virtual environment

If you wish to use `magenpy` on a shared computing environment or cluster, it is recommended that you install 
the package in a virtual environment. Here's a quick example of how to install `magenpy` on a SLURM-based cluster:

```bash
module load python/3.8
python -m venv magenpy_env
source magenpy_env/bin/activate
python -m pip install --upgrade pip
python -m pip install magenpy>=0.1
```

### Using `Docker` containers

If you are using `Docker` containers, you can build a container with the `viprs` package 
and all its dependencies by downloading the relevant `Dockerfile` from the 
[repository](https://github.com/shz9/magenpy/tree/master/containers) and building it 
as follows:

```bash
# Build the docker image:
docker build -f cli.Dockerfile -t magenpy-cli .
# Run the container in interactive mode:
docker run -it magenpy-cli /bin/bash
# Test that the package installed successfully:
magenpy_ld -h
```

We plan to publish pre-built `Docker` images on `DockerHub` in the future.