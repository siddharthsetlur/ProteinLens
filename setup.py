from setuptools import find_packages, setup

setup(
    name="proteinlens",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "protein_results": ["*.py"],
        "protein_results.geometry": ["*.py"],
        "proteinlens.viz": [
            "static/*",
            "static/geopedia/*",
            "static/js/*",
            "static/css/*",
        ],
    },
    install_requires=[
        "biopython>=1.84",
        "fastapi>=0.115",
        "httpx>=0.27",
        "joblib>=1.4",
        "numpy>=1.26,<2",
        "orjson>=3.10",
        "pandas>=2.0",
        "pyyaml>=6",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "uvicorn>=0.30",
    ],
    entry_points={"console_scripts": ["geopedia=proteinlens.viz.server:main"]},
)
