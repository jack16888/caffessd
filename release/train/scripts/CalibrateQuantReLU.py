""" 
All Rights Reserved.

Copyright (c) 2017-2019 Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,                  
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
This script contains a class with multiple functions that would be used to turn on ReLU quantization

### Class name: CalibrateQuantReLU ###
  --Usage: Used after training with quantized convolution layers and before training with quantized ReLU layers. The resulting network will be used for initialization in the QuantReLU finetuning.

  --Details: Estimate the output range of each ReLU activation layers to initialize the QuantReLU alpha values 

  --Functions: 
    (1) EstimateQuantReLU
        Estimate the output range of each ReLU layers by inferencing over a image batch. 
    (2) RefreshModel 
        Update .prototxt and .caffemodel with the estimated ReLU alpha values
 
"""

import os
import sys
import numpy as np
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser


class CalibrateQuantReLU():  
    def __init__(self, prototxt, model):
        
        self.prototxt = prototxt
        self.model = model
        caffe.set_mode_gpu() 
        self.net = caffe.Net(prototxt, model, caffe.TEST)
        self.ReLUAlpha = []
        self.quant_enable = True
        self.log_file = "ReLUAlpha.log" 
        
    def EstimateQuantReLU(self, num_batch=10, percentile=0.999):

        layer_out_dict= {}
        for i in range(num_batch):
            self.net.forward()
            for idx, layer in enumerate(self.net.layers):
                layer_name = self.net._layer_names[idx]
                layer_top = self.net.top_names[layer_name][0]
                layer_type = layer.type
                if layer_type == "QuantConvolution":
                   layer_out = np.array(self.net.blobs[layer_top].data[...])
                   layer_out = layer_out.flatten()
                   if layer_name in layer_out_dict.keys():
                      layer_out_acc = layer_out_dict[layer_name]
                      layer_out_dict[layer_name] = np.concatenate((layer_out_acc, layer_out))
                   else:
                      layer_out_acc = layer_out
                      layer_out_dict[layer_name] = layer_out_acc 

        print('Layer Name\t\tCount\t\tEstimated QuantReLU Alpha (@ {}% percentile)'.format(percentile*100))                                                                  

        for idx, layer in enumerate(self.net.layers):
            layer_name = self.net._layer_names[idx]
            if self.log_file:
               with open(self.log_file, 'a+') as f:
                 if idx==0:
                    f.write('Layer name' + '\t' + 'Count' +'\t\t' + 'QuantReLU value' + '\n')
            if layer_name in layer_out_dict.keys():                  
               count = len(layer_out_dict[layer_name]) 
               target_idx = int(count * percentile) 
               layer_out_dict[layer_name].sort()                
               self.ReLUAlpha.append(layer_out_dict[layer_name][target_idx])

               print('{}\t\t\t{}\t\t\t{}'.format(layer_name, count, layer_out_dict[layer_name][target_idx]))
               if self.log_file:
                  with open(self.log_file, 'a+') as f:
                    f.write(layer_name + '\t\t' + str(count) + '\t' +  str(layer_out_dict[layer_name][target_idx]) + '\n') 
 
    def RefreshModel(self):

        with open(self.prototxt) as f:
            net_model = caffe.proto.caffe_pb2.NetParameter()
            pb.text_format.Merge(f.read(), net_model)
            relu_count = 0
            for i, layer in enumerate(net_model.layer):           
                if layer.type == 'QuantReLU':
                   layer.param[0].lr_mult = 0.0 
                   layer.quant_relu_param.filler.value = self.ReLUAlpha[relu_count]
                   layer.quant_relu_param.quant_enable = True
                   relu_count += 1 
       
        prototxt_QuantReLU = os.path.splitext(self.prototxt)[0] + '_QuantReLU.prototxt'  
        model_QuantReLU = os.path.splitext(self.model)[0] + '_QuantReLU.caffemodel'  
        with open(prototxt_QuantReLU, 'w') as f:
           f.write(pb.text_format.MessageToString(net_model))

        relu_count = 0
        for i, layer in enumerate(self.net.layers):
            layer_name = self.net._layer_names[i] 
            if layer.type == "QuantReLU":
	       self.net.params[layer_name][0].data[...] = self.ReLUAlpha[relu_count]
               relu_count += 1

        self.net.save(model_QuantReLU)       


if __name__ == '__main__':

   prototxt = "/home/zhangwanchun/caffe-ssd/step3/VGG_SSD_224_5801_train_quant_org.prototxt"   # The network prototxt from QuantConvTraing
   model = "/home/zhangwanchun/caffe-ssd/step3/pretrain_iter_195000.caffemodel"    # The network caffemodel from QuantConv Training
   num_batch = 10
   percentile = 0.999

   net = CalibrateQuantReLU(prototxt, model)
   net.EstimateQuantReLU(num_batch, percentile)
   net.RefreshModel() 

