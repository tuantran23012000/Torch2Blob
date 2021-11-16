import sys
import torch.onnx
import torch
import argparse
from torch.autograd import Variable
sys.path.append(".")
import os
from utility import get_kernel, parse_model_name
from model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE

parser = argparse.ArgumentParser()
parser.add_argument( "--in_model", type=str,required=True) # input model name
parser.add_argument( "--out_model", type=str,required=True)  # output model name
parser.add_argument( "--h",type=int,required=True) # height_image
parser.add_argument( "--w",type=int,required=True) # width_image
args = parser.parse_args()

model_path=os.path.join('./models/',args.in_model)
output_model_name=args.out_model
MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}
model_name = os.path.basename(model_path)
h_input, w_input=[args.h,args.w]
_, _, model_type, _ = parse_model_name(model_name)
kernel_size = get_kernel(h_input, w_input,)
device = torch.device("cuda:{}".format(0)
                                   if torch.cuda.is_available() else "cpu")
model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

dummy_input = Variable(torch.randn(1, 3, h_input, w_input))
shape=[1,3,h_input,w_input]
state_dict = torch.load(model_path,map_location=device)
keys = iter(state_dict)
first_layer_name = keys.__next__()
if first_layer_name.find('module.') >= 0:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name_key = key[7:]
        new_state_dict[name_key] = value
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)
print("Success load model !")
torch.onnx.export(model,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  output_model_name+".onnx",   # where to save the model (can be a file or file-like object)
                  #export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  #do_constant_folding=True,  # whether to execute constant folding for optimization
                  #input_names = ['data'],   # the model's input names
                  #output_names = ['output'], # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #              'output' : {0 : 'batch_size'}})
)
print("Success export model.onnx !")
print("input shape image : ",shape)
print("path your model.onnx :",os.path.join(os.getcwd(),output_model_name+".onnx"))
print("path your folder output model.bin and model.xml :",os.getcwd()+"\models_output")