Disclaimer: This GitHub repository is under routine maintenance.

# FASTEN

This is the source code for the paper "Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection" (NeurIPS 2023). Our paper proposes a novel 3D mask detection framework, called FASTEN (Flow-Attention-based Spatio-Temporal aggrEgation Network), which only requires five frames of input.  Our proposed network contains three key modules: 1) a facial optical flow network to obtain non-RGB inter-frame flow information; 2) flow attention to assign different significance to each frame; 3) spatio-temporal aggregation to aggregate high-level spatial features and temporal transition features.

## Setup

We implement this repo with the following environment:
- Ubuntu==20.04
- Python==3.8
- Pytorch==1.12.1
- CUDA==12.0

Install other packages via:

``` bash
pip install -r requirements.txt
```

### Dataset

The `CASIA-SURF HiFiMask` dataset can be download from the [Official Website of CASIA-SURF HiFiMask](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-hifimaskiccv2021). 

### Pretrained model
The pre-trained model for HiFiMask is provided [here](https://1drv.ms/u/s!Aj2hSJitqRWpgVbUtXkGZjhAx_ey).

## Usage

Inference code is provided, you need to modify `/data/test_hifi.lst` to adapt to your own path.

* Replace the `/PATH/TO/HIFIMASK/` in `test_hifi.lst`  with the path where you store the HiFiMask dataset.
* Modify `test_path` or `resume`  in  `/experiment/face.py` if needed.
* run `python main.py`.

## Acknowledgement

Part of our implementation is built on [MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch) and [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch). Thanks for their extraordinary works!

## License 
This software and associated documentation files (the "Software"), and the research paper (Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection) including but not limited to the figures, and tables (the "Paper") are provided for academic research purposes only and without any warranty. Any commercial use requires our consent. Please contact sanerzz@outlook.com if needed.
