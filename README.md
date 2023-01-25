# Install Environment on Anaconda
- Issue: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'


[Solution](https://stackoverflow.com/questions/61915607/commandnotfounderror-your-shell-has-not-been-properly-configured-to-use-conda): 
In terminal
```
conda create -n NAME
conda remove --name NAME --all
activate NAME
conda deactivate

conda env list
conda info --envs
source ~/opt/anaconda3/etc/profile.d/conda.sh   ## option
## if happens: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
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

# Install Spyder
```
conda install -c anaconda spyder
```

## [Unable to launch Spyder after installation of PyQt5](https://stackoverflow.com/questions/63409417/unable-to-launch-spyder-after-installation-of-pyqt5)
Solution：Uninstall pyqt5(pip uninstall pyqt5) and then reinstall it version 5.11.3(pip install pyqt5==5.11.3) After installing the new version it'll show an error msg but it will work anyways. This has worked for me and my colleges.

# Issue ''numpy.ndarray' object has no attribute 'append''

Solution
```
conda install numpy=1.20
```

# Check python version in Jupyter notebook
```
from platform import python_version
print(python_version())
```

# Install torch 1.8 on Wins10
- Check or update Nvidia GPU driver version. Here is the corresponding CUDA toolkit list [Reference](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 
- Download CUDA Toolkit 10.2 version [Reference](https://developer.nvidia.com/cuda-10.2-download-archive) 

CMD line
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
- Based on the instruction to choose torch package version ([Link1](https://pytorch.org/get-started/previous-versions/))

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
  - Install solution (the best one!!!!): [Reference](https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c)
## VS on wins
- Microsoft Visual C++ Redistributable Latest Supported: [Download](https://docs.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170)
  - Install Microsoft Visual C++ 2015-2019 Redistributable (x64)&(x86)
  -  Microsoft Visual Studio 2019


# Install Mask R-CNN 
[Reference](https://github.com/matterport/Mask_RCNN)

```
conda create -n maskrcnn python=3.7.6
conda activate maskrcnn

pip install numpy
pip install scipy
pip install Pillow
pip install cython
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install imgaug
pip install IPython
```
```
conda install tensorflow==1.14.0
conda install keras
conda install tensorflow-gpu  ## for wins
conda install h5py==2.10.0
pip install keras_applications==1.0.7
```

Load python
```python
import tensorflow as tf
import keras
import h5py

print(tf.__version__)
print(keras.__version__)
print(h5py.__version__)

## Check whether GPU is being or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

```
Show
```
1.14.1
2.3.1

[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 6757537999743423464
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 7109751604
locality {
  bus_id: 1
  links {
  }
}
incarnation: 6561882754128707177
physical_device_desc: "device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1"
]
```

## Issue
Description:Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. #24828

Alternative description: Delete the underlying status object from memory otherwise it stays alives there is a reference to status from this from the traceback due to UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [[{{node sequential_1_1/Conv1/convolution}}]] [[{{node loss/add_7}}]]

- **Solution (Best)**: restart the computer
- II
```python
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```
- III
```
pip uninstall tensorflow-gpu
conda install tensorflow-gpu
```

# Install Hotspotter 
## Requirment
[Reference](https://github.com/SU-ECE-18-7/hotspotter)
- MacOXS
```
pip install pyhesaff ## only support Linux & MacOXS
```



