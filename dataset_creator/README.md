
# Description

Experiments with CLIP based image search

# Setup and usage

```bash
cd ./dataset_creator/
# create conda environment
conda env create -f environment.yml
# install this pkg
pip install -e .
```

```bash
# then it should be possible to run existing scripts
python scripts/download_data.py
python scripts/create_dataset.py
```

# Ideas / TODOs

- [x] script to download files from internet (Pixabay API)
- [x] CLIP based image directory search
  - [x] image based search
  - [x] text based search
  - [ ] diversity sampling
- GT generation
  - [x] image captions (based on COCA model)
  - [ ] BBox (based on Grounding-Dino)
  - [ ] SemSeg (based on Grounded-SAM)

# References

<https://github.com/mlfoundations/open_clip>
<https://github.com/IDEA-Research/Grounded-Segment-Anything>
<https://medium.com/red-buffer/diving-into-clip-by-creating-semantic-image-search-engines-834c8149de56>
