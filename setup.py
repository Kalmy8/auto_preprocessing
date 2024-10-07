from os.path import abspath, dirname, join

import toml
from setuptools import find_packages, setup

pyproject_path = join(dirname(abspath("__file__")), "../pyproject.toml")
file = open(pyproject_path, "r")
toml_str = file.read()
parsed_toml = toml.loads(toml_str)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="prepCV",
    version=parsed_toml["tool"]["commitizen"]["version"],
    description="Define preprocessing pipelines using nested callable functions and parameter grids for them.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kalmy8",
    author_email="kalmykovalexey01@gmail.com",
    url="https://github.com/Kalmy8/auto_preprocessing",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy~=1.21.4",
        "matplotlib~=3.5.1",
        "opencv-python",
        "dill",
    ],  # List dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Specify minimum Python version
)
