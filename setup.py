# Author: Kobie Kirven
# Davenport Lab - Penn State University
# Date: 9-2-2021

# Updated by: Hanh Tran
# Date updated: 9-5-2024

import setuptools
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

setuptools.setup(
    name="scims",
    version="1.0.0", # new version of SCiMS using Bayesian approach
    packages=["scims", "scims"],
    include_package_data = True,
    package_data={
        'scims': ['training_data/training_data.txt'],
    },
    entry_points={"console_scripts": ["scims=scims.__main__:main",],},
    description="SCiMS: Sex Calling for Metagenomic Sequences",
    install_requires=["setuptools", "biopython", "pandas", "matplotlib", "scipy", "numpy"],
    python_requires=">=3.6",
)