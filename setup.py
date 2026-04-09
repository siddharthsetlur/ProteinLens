from setuptools import find_packages, setup

setup(
    name="proteinlens",
    version="1.0.0",
    packages=find_packages(),
    package_data={"protein_results": ["*.py"], "protein_results.geometry": ["*.py"]},
)
