from pathlib import Path  # noqa: INP001

from setuptools import setup

here = Path.resolve(Path(__file__).parent)

with Path("README.md").open(encoding="utf-8") as freadme:
    long_description = freadme.read()

setup(
    name="industrial_detection",
    version="0.1",
    description="Solving industrial detection tasks for pick and place robots in industrial settings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManuelPalermo",
    author_email="macpalermo@gmail.com",
    url="https://github.com/ManuelPalermo/dev-playground/tree/main/industrial_detection",
    packages=["industrial_detection"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
