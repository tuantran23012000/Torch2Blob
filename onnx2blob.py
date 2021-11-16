import blobconverter
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument( "--xml_model", type=str,required=True) # input model name
parser.add_argument( "--bin_model", type=str,required=True)  # output model name

args = parser.parse_args()

blob_path = blobconverter.from_openvino(
    xml=args.xml_model,
    bin=args.bin_model,
    data_type="FP16",
    shaves=4,
)