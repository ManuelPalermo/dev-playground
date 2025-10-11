
# Traffic Monitoring System

This project implements a basic Traffic Monitoring System using a fixed stationary camera with primary purpose of traffic flow monitoring in realtime with the following components:

- Vehicle detection
- Vehicle tracking
- Vehicle speed estimation

It contains the following features:

- A vehicle detection system and count the number of incoming and outgoing vehicles. With support for additional attributes (sedan, SUV etc. not trucks and vans).
- Track vehicles with a consistent id between frames. And support simple future state prediction based on tracking model.
- Estimation of the real world velocity (km/h) of the vehicles based on perspective transform (from camera to bev) and tracking.

<https://github.com/user-attachments/assets/1308cf27-b977-4de8-952b-bd91ee15a226>

## Setup

```bash
cd ./traffic_detection/
# create conda environment
conda env create -f environment.yml
# install pkg
pip install -e .
```

## Run script

```bash
python scripts/main.py \
  --source ./data/Video.mp4 \
  --artefacts_output_dir ./outputs/ \
  --num_frames 2400 \
  --detector_name "yolov10x_onnx"
  --visualize_frames True \
  #--save_frames_to_disk True \
```

## Run tests

```bash
# run pytest with coverage
pytest ./tests/ \
  --cov=./traffic_detection \
  --cov=./scripts/ \
  --cov-branch \
  --cov-report=term-missing
```

---

### Current status

- [x] 2D detector
  - [x] real time ONNX model
  - [x] HugginFace pretrained models
  - [x] HugginFace VLM models with promptable classes (slower)
  - [x] NMS (based on box IoU)
  - [ ] Search for more pretrained models with improved runtime/detections and desired classes (RF-DETR, YOLOv11, etc...)

- [x] Multi-Object Tracker (SORT + Kalman)
  - [x]  Consistent tracks + control for spawning/killing tracks
  - [x] Estimate boxes velocity
  - [x] Predict next n states
  - [ ] Initialize speed faster (dont wait for filter, just take first few measurements for first speed estimate)
  - [ ] Track in BEV space (closer to linear) (or use non-linear filter)
    - [ ] Get velocity from tracker in BEV space
  - [ ] Use better association metric (IoU bad for small objects, also not possible in BEV since no BevBox, e.g. use center distance instead)
  - SORT tracker, average out properties over n frames (confidence, class, color, etc...)
  - [ ] Tune tracker or try SOTA trackers (e.g. DeepSORT, ByteTrack, etc...)

- [x] Object Properties Inference:
  - [x] Takes the detection box patch and computes features
    - [x] Object Color (from average pixels color)
    - [ ] Try CLIP on the image patches to compute any desired property

- [x] Counter for vehicles of interest
  - [x] Count only vehicles inside areas
  - [x] Count only vehicles with desired properties
  - [ ] Count on a small patch to reduce double counting

- [x] Estimate objects' real world position (Perspective transform based on "known" road markings distance)
  - [x] Compute real world velocity
  - [x] Determine the fastest car in a given frame (purple instead of green)
  - [ ] Support curved road geometries (e.g. distance based on curved polylines (known lane markings))
  - [ ] Automatically estimate ground plane, and areas of interest (less manual tuning to specific scene)

- [ ] Tech Debt 😿
  - [x] Add integration test
  - [x] Add some unittests
  - [x] Fix SCA
  - [x] Cleanup documentation
  - [ ] Refactor some classes with mixed responsabilities
