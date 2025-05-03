from pathlib import Path  # noqa: INP001

from setuptools import setup

here = Path.resolve(Path(__file__).parent)

with Path("README.md").open(encoding="utf-8") as freadme:
    long_description = freadme.read()

setup(
    name="llm_app",
    version="0.1",
    description="LLM chat app.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManuelPalermo",
    author_email="macpalermo@gmail.com",
    url="https://github.com/ManuelPalermo/dev-playground/tree/main/dataset_creator",
    packages=["llm_app"],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
