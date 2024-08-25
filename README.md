# Dev Playground

Simple containerized environment to quickly try random stuff, with a focus on python and deep learning tools.

Also includes some [experiments and results](#explored-topics).

## Features

- Dockerfile:
  - ubuntu22.04
  - nvidia-cuda12.6
  - cpp tools
  - python tools
  - shell tools
  - xauth/X11
  - miniconda
- vscode configs (vscode/settings.json and devcontainer.json)
  - gpu support
  - graphical interface support (X11)
  - host camera support
- Python tool configs (linters, formatters)
- Pre-commit hooks
- Basic CI/CD (github actions) for python formatting/linting

---

## Explored topics

- **[diffusion](./diffusion/README.md)**: data generation using DDPMs (conditional and unconditional generation).

- **[dataset_creator](./dataset_creator/README.md)**: CLIP based text and image search over an image directory and autolabeling using foundational models.

- **dataset_selection**: Selecting of most diverse images of a directory based on similarity algorithm (classical CV).
