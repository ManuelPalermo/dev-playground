from os import path
import itertools

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open("README.md", "r") as freadme:
    long_description = freadme.read()

setup(
    name="ddpm",
    version="0.1",
    description="DDPMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManuelPalermo",
    author_email="macpalermo@gmail.com",
    url="https://github.com/ManuelPalermo/dev-playground/tree/main/ddpm",
    packages=["ddpm"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
