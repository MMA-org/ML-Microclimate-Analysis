# Installation

This document provides instructions for setting up and installing the project on your local machine.

## Prerequisites

Before installing the project, ensure that you have the following prerequisites installed on your system:

- **Python**: Version 3.8 or higher
- **pip**: The Python package manager
- **Virtual Environment** (optional but recommended): `venv`, `virtualenv`, or a similar tool
- Any other system requirements (e.g., specific libraries, tools, or dependencies)

## Installation Steps

### 1. Clone the Repository

Clone the project repository from your version control system:

```bash
git clone https://github.com/MMA-org/ML-Microclimate-Analysis
cd ML-Microclimate-Analysis
```

### 2. Set Up a Virtual Environment

(Optional but recommended) Create and activate a virtual environment to isolate the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate   # On Windows
```

### 3. Install the Project

Use `pip` to install the project and its dependencies from `pyproject.toml`:

```bash
pip install .
```

For development purposes:

```bash
pip install -e .[dev]
```

## Verifying Installation

Run the following command to ensure the project is correctly installed:

```bash
lcm --help
```

This should display the project's CLI usage or help information.

## Using the Project

After installation, you can begin using the project. Refer to the relevant sections in the documentation for usage instructions.
