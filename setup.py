from setuptools import setup, find_packages

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="prepCV",
    version = "0.0.1",
    description = "Define preprocessing pipelines using nested callable functions and parameter grids for them.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = "Kalmy8",
    author_email="kalmykovalexey01@gmail.com",
    url="https://github.com/Kalmy8/auto_preprocessing",
    packages=find_packages(exclude=["tests*"]),
    install_requires=['numpy~=1.21.4', 'matplotlib~=3.5.1', 'opencv-python'],  # List dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9', # Specify minimum Python version
)



