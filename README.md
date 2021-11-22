# Install Environment on Anaconda
## Issue: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate' 
[Solution](https://stackoverflow.com/questions/61915607/commandnotfounderror-your-shell-has-not-been-properly-configured-to-use-conda): 
In terminal
```
conda create -n NAME
conda remove --name NAME --all
activate NAME
deactivate

conda env list
conda info --envs
source ~/opt/anaconda3/etc/profile.d/conda.sh   ## option
conda activate Detectron
```
Show
```
# conda environments:
#
base                     /Users/obaiga/opt/anaconda3
Detectron             *  /Users/obaiga/opt/anaconda3/envs/Detectron

```
# Install a list of Python package from a yam file 
Example
```
name: detectron2
channels:
  - pytorch
  - conda-forge
  - anaconda
  - defaults
dependencies:
  - python=3.8
  - numpy
  - pywin32
  - cudatoolkit=11.0
  - pytorch==1.7.1
  - torchvision
  - git
  - pip
  - pip:
    - git+https://github.com/facebookresearch/detectron2.git@v0.3
```
[Reference](https://stackoverflow.com/questions/54492671/how-to-install-list-of-python-libraries-using-yml-file-without-making-new-enviro)
```
## You can use the conda env update command:
conda env update --name <your env name> -f <your file>.yml

## or, if the environment you want to update is already activated, then
conda env update -f <your file>.yml
```

# Check python version in Jupyter notebook
```
from platform import python_version
print(python_version())
```

# Install torch 1.8 on Wins10
1. Check or update Nvidia GPU driver version. Here is the corresponding CUDA toolkit list [Reference](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Download CUDA Toolkit 10.2 version [Reference](https://developer.nvidia.com/cuda-10.2-download-archive) 

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

```
## Wins
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

## MacOSX
conda install pytorch torchvision torchaudio -c pytorch
```
```python 
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
# Install Colaboratory
```
conda install -c conda-forge google-colab
```

# Install Detectron2
[Detectron2](https://github.com/facebookresearch/detectron2)
## Requirement
- Wins10 [Reference](https://github.com/facebookresearch/detectron2/issues/9)
  - Solution: [Reference](https://medium.com/@yogeshkumarpilli/how-to-install-detectron2-on-windows-10-or-11-2021-aug-with-the-latest-build-v0-5-c7333909676f)

```
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
```
Show
```
** Visual Studio 2019 Developer Command Prompt v16.3.1
** Copyright (c) 2019 Microsoft Corporation
```

