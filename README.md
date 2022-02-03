# cpp-torch
Template sources for writing custom C++ extensions of PyTorch.

Implementations are based on 
- https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/mkldnn
- https://github.com/liangfu/tiny-cnn


## Usage instructions  
- Move into one of the folders in `src/`.
- Execute `python setup.py install` to build the C++ extension.
- Check the result of the operation by running `python test.py`.
