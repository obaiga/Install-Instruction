# Install torch on Wins10
1. Check or update Nvidia GPU driver version. Here is the corresponding CUDA toolkit list [Link](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Download CUDA Toolkit 10.2 version [Link](https://developer.nvidia.com/cuda-10.2-download-archive) 

In cmd
```
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 441.22       Driver Version: 441.22       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080   WDDM  | 00000000:01:00.0  On |                  N/A |
| 74%   79C    P2   166W / 200W |   5213MiB /  8192MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+

nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
Cuda compilation tools, release 10.2, V10.2.89

```


Based on the instruction to choose torch package version ([Link1](https://pytorch.org/get-started/previous-versions/))

```python
 pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
 
 
import torch, torchvision
print(torch.__version__)
print(torch.cuda.get_device_name())
print(torch.version.cuda)
print(torch.cuda.is_available())
x = torch.randn(1).cuda()
print(x)
```
Show
```
1.8.1+cu102
GeForce GTX 1080
10.2
True
tensor([0.7653], device='cuda:0')
```

# Install Detectron2
[Detectron2](https://github.com/facebookresearch/detectron2)
## Requirement
-**Linux or macOS with Python ≥ 3.6, NO WINS**
-**PyTorch ≥ 1.8** and torchvision that matches the PyTorch installation. 
-OpenCV is optional but needed by demo and visualization
