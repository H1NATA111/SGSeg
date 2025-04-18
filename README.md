# "Soft-Guided Open-Vocabulary SemanticSegmentation of Remote Sensing Images"
![](fig/1.png)

## Comparison With State-of-the-art Methods
![](fig/2.png)

## Installation

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

## Training script
```bash sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]
# For example
sh run.sh configs/vitl_336.yaml 4 output/```


## Evaluation script
```bash sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]```


## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [CLIP](https://github.com/openai/CLIP.git),[OVRS](https://github.com/caoql98/OVRS.git),[CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg.git),[SED](https://github.com/xb534/SED).
