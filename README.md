# Dev Playground

Simple development playground to quickly try random stuff. Features include:

- Dockerfile:
  - ubuntu22.04
  - nvidia-cuda11.7
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

# Explored topics

- **diffusion**: data generation using DDPMs (conditional and unconditional generation).

- **dataset_creator**: CLIP based text and image search over an image directory and image autolabelling.

- **dataset_selection**: Selecting of most diverse images of a directory based on similarity algorithm (classical CV).
