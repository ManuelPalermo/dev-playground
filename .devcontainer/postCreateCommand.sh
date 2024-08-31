#!/usr/bin/bash

set -eo pipefail

HOME=/home/${USER}
#WS_PATH="${HOME}/dev-playground"
#cd "${WS_PATH}"
#sudo chown -R "${USER}" "${HOME}"

echo ">> Updating apt-get package list"
sudo apt-get update

echo ">> Setting up miniconda"
source "${HOME}/miniconda3/bin/activate"
conda init bash
conda deactivate
conda config --set auto_activate_base false
conda update -n base -c defaults conda -y

echo ">> Installing pre-commit hooks"
pre-commit install && pre-commit autoupdate
