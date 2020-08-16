#-*- coding:utf-8 -*-

import os
import sys
import onnx
import onnxruntime
import numpy as np
import torch
from models import Darknet

def transform_to_onnx(cfg_file, weight_file, batch_size, in_h, in_w):
    model = Darknet(cfg_file)
    pre_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(pre_dict['model'])
    x = torch.ones((batch_size, 3, in_h, in_w), requires_grad=True)*120 /255.0
    onnx_file_name = 'model/yolov3.onnx'
    
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      #export_params=True,
                      #opset_version=11,
                      #do_constant_folding=True,
                      input_names=['input'], output_names=['output1'])
                      #dynamic_axes=None)
    print('Onnx model exporting done')
    return onnx_file_name, x

def main(cfg_file, weight_file, batch_size, in_h, in_w):
    onnx_path_demo, x = transform_to_onnx(cfg_file, weight_file, 1, in_h, in_w)
    #session = onnxruntime.InferenceSession(onnx_path_demo)
    #output = session.run(['output1'], {'input':x.detach().numpy()})


if __name__ == "__main__":
    cfg_file = 'cfg/yolov3-tiny.cfg'
    weight_file = 'model/last.pt'
    batch_size = 1
    in_h = 416
    in_w = 416
    main(cfg_file, weight_file, batch_size, in_h, in_w)
