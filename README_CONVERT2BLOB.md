# CONVERT YOUR MODEL TO IR FORMAT

## INSTALLATION
- OpenVino Toolkit 

  Download openvinotoolkit 2021.3 version : [Link](https://registrationcenter-download.intel.com/akdlm/irc_nas/17662/l_openvino_toolkit_p_2021.3.394.tgz)

  Instruction of installation : [Link](https://docs.openvinotoolkit.org/latest/installation_guides.html)
- Pytorch  
- Torchvision
- Blobconverter

## PYTORCH2ONNX

Create file pytorch2onnx.py and edit follow below :

- Import packages :
  ```python
  import sys
  import torch.onnx
  import torch
  import argparse
  from torch.autograd import Variable
  sys.path.append(".")
  import os
  ```

- Define image shape :

    ```python 
    dummy_input = Variable(torch.randn(1, num_channels, h_image_input, w_image_input))
    ```

- Load model :

  ```python 
  state_dict = torch.load(model_path,map_location='cpu' or 'gpu')
  ```

- Export model :

  ```python
  model.load_state_dict(state_dict)
  torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  "model_name_output.onnx",   # where to save the model (can be a file or file-like object)
                  #export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  #do_constant_folding=True,  # whether to execute constant folding for optimization
                  #input_names = ['data'],   # the model's input names
                  #output_names = ['output'], # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #              'output' : {0 : 'batch_size'}})
    )
  ```

Detail for Instruction : [Link](https://pytorch.org/docs/stable/onnx.html#id2)

## ONNX2IR

Complete installation openvinotoolkit, Cd ../openvino_2021/deployment_tools/model_optimizer and run :

  ```python    
  python mo.py --input_model <path your Model.onnx> --input_shape [1, num_channels, h_image_input, w_image_input] --output_dir <path folder for your Model.bin and Model.xml>
  ```

## IR2BLOB

- Using online converter : [Link](http://luxonis.com:8080/)

  Choose openvino version (default 2021.3 version), next step choose OpenVino Model then upload Model.bin and Model.xml

- Or using package blobconverter : [Link](https://pypi.org/project/blobconverter/)

## Documentation

[Documentation](https://linktodocumentation)

## Example

[README_Example.md](./README_Example.md)
