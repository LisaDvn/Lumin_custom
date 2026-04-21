# Installation

## Requirements

- Python 3.10+
- pip

## Install

Open **Anaconda Prompt** and follow the steps below.

**1. Clone the repository**

```bash
git clone https://github.com/LisaDvn/Lumin_custom.git
```

**2. Navigate to the cloned folder**

```bash
cd Lumin_custom
```

**3. Create a conda environment**

```bash
conda create -n lumin_env python=3.10 cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge -y
```

**4. Activate the conda environment**

```bash
conda activate lumin_env
```

**5. Install the package and all dependencies**

```bash
pip install -e .
```

## Run the NAPARI, LUMIN is now a plugin

```bash
NAPARI
```