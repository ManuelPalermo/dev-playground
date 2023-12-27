
# Setup and usage

```bash
cd ./diffusion/
# create conda environment
conda env create -f environment.yml
# install this pkg
pip install -e .
```

```bash
# then it should be possible to run existing scripts with configs (hydra confs based)
python scripts/train.py --config-name "MNIST"
```

NOTE: check the `./config/` folder to see existing configs. Any param can also be overwritten from the cli.

# Ideas / TODOs

- [x] calculate FID
- [x] conditional generation
- [x] add model EMA
- [x] add option to use custom folder with data (also util to overfit on the data from folder)
- [ ] add generation of 2D/3D pointclouds
- [ ] "pixel/point" conditional generation (e.g. based on semseg masks)
- [x] add hydra conf

# Aknowledgments

Based on projects from:

- <https://learnopencv.com/denoising-diffusion-probabilistic-models/>
- <https://github.com/filipbasara0/simple-diffusion>
- <https://github.com/lucidrains/denoising-diffusion-pytorch>
- <https://github.com/hojonathanho/diffusion>
- <https://github.com/openai/improved-diffusion>
- <https://github.com/luost26/diffusion-point-cloud>
