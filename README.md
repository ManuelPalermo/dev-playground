# Dev Playground

Simple containerized environment to quickly try random stuff, with a focus on python and deep learning tools.

Also includes some [experiments and results](#explored-topics).

## Features

- Dockerfile:
  - ubuntu24.04
  - nvidia-cuda12.6
  - cpp tools
  - python tools
  - node/js tools
  - shell tools
  - xauth/X11
  - miniconda
- vscode configs (vscode/settings.json and devcontainer.json)
  - gpu support
  - graphical interface support (X11)
  - host camera support
- Pre-commit hooks
- Python tool configs (linters, formatters)
- Node/js tool configs (linters, formatters)
- CI/CD (github actions) for formatting/linting

---

## Explored topics

- **[devcontainer](.devcontainer/)**: containerized docker environment with dev tools and configs.

- **[diffusion](./diffusion/README.md)**: data generation using DDPMs (conditional and unconditional generation).

- **[dataset_creator](./dataset_creator/README.md)**: CLIP based text and image search over an image directory and autolabeling using foundational models.

- **[llm_app](./llm_app/README.md)**: LLMs + Retrieval Augmented Generation (RAG) with a Web chat interface (similar to openAI but worse).
 
- **dataset_selection**: Selecting of most diverse images of a directory based on similarity algorithm (classical CV).
