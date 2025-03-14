from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "geqmd/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="geqmd",
    version=version,
    description="GEqMD is a software tool to run MD simulations using classical force fields jointly with pytorch EGNN FFs",
    download_url="https://github.com/limresgrp/GEqMD",
    author="Daniele Angioletti",
    python_requires=">=3.8",
    packages=find_packages(include=["geqmd", "geqmd.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
        ]
    },
    install_requires=[
        "einops",
    ],
    zip_safe=True,
)
