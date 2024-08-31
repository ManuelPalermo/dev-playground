from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open("README.md", encoding="utf-8") as freadme:
    long_description = freadme.read()

setup(
    name="ds_creator",
    version="0.1",
    description="Dataset Search and Annotation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManuelPalermo",
    author_email="macpalermo@gmail.com",
    url="https://github.com/ManuelPalermo/dev-playground/tree/main/dataset_creator",
    packages=["ds_creator"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
