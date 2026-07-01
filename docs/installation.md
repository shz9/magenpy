The `magenpy` software is written in `Cython/Python3`.
The software is designed to be used in a variety of computing environments, including local workstations, 
shared computing environments, and cloud-based computing environments. Because of the dependencies on `Cython`, you need 
to ensure that a `C/C++` Compiler (with appropriate flags) is present on your system.

## Requirements

Building the `magenpy` package requires the following dependencies:

* `Python` (>=3.10)
* `C/C++` Compiler
* `Cython` (>=0.29.21)
* `NumPy` (>=2.0)

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
python -m pip install magenpy
```

### Using `uv`

If you use [`uv`](https://docs.astral.sh/uv/) to manage Python environments, you can install `magenpy` into the
current environment with:

```bash
uv pip install magenpy
```

You can also create a dedicated virtual environment first:

```bash
uv venv --python 3.11 magenpy_env
source magenpy_env/bin/activate
uv pip install magenpy
```

For command-line use, `uvx` can run the installed console scripts in an isolated temporary environment:

```bash
uvx --from magenpy mgp_compute_ld -h
uvx --from magenpy mgp_simulate -h
```

This is useful on systems where you want to try the command-line tools without permanently installing `magenpy`
into your active Python environment.

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
module load python/3.11
python -m venv magenpy_env
source magenpy_env/bin/activate
python -m pip install --upgrade pip
python -m pip install magenpy
```

### Using `Docker` containers

If you are using `Docker` containers, you can build a CLI image with the `magenpy` package, PLINK/PLINK2,
and the standard command-line tools:

```bash
# Build the linux/amd64 image from the local repository checkout:
docker build --platform linux/amd64 -f containers/cli.Dockerfile -t magenpy-cli .

# Run one of the command-line tools:
docker run --rm magenpy-cli mgp_compute_ld -h
docker run --rm magenpy-cli mgp_simulate -h
docker run --rm magenpy-cli mgp_extract_ld -h
docker run --rm magenpy-cli mgp_prune_ld -h
docker run --rm magenpy-cli mgp_expand_ld -h
```

To build the image from the latest PyPI release instead of the local checkout, pass the PyPI install target:

```bash
docker build --platform linux/amd64 -f containers/cli.Dockerfile \
  --build-arg MAGENPY_INSTALL_TARGET=magenpy \
  -t magenpy-cli .
```

Once a DockerHub image is published, you can run the same commands by replacing `magenpy-cli` with the
published image name, for example:

```bash
docker run --rm shadizabad/magenpy:latest mgp_compute_ld -h
```

### Using `Apptainer`

On shared computing clusters where Docker is not available, you can run the command-line tools with
[`Apptainer`](https://apptainer.org/) once the Docker image is live on DockerHub. Apptainer can pull and
run Docker images directly:

```bash
apptainer run docker://shadizabad/magenpy:latest mgp_compute_ld -h
apptainer run docker://shadizabad/magenpy:latest mgp_simulate -h
apptainer run docker://shadizabad/magenpy:latest mgp_extract_ld -h
```

For repeated use, pull the image into a local SIF file:

```bash
apptainer pull magenpy_latest.sif docker://shadizabad/magenpy:latest
apptainer run magenpy_latest.sif mgp_compute_ld -h
```
