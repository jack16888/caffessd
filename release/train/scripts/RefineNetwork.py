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
This script contains a class with multiple network processing functions that would be used during GTI quantization-aware training

## Class name: RefineNetwork ###
   --Usage: Used after training with "QuantConvolution + QuantReLU" and before the last fine-tuning step. The resulting network will be used for initialization in the last round of finetuning.

   --Details: Merge batch normalization layers into corresponding convolution layers,if applicable
              Merge input mean values into biases of the first QuantConvolution layer, if applicable
              Equalize the input and output range of each layer to [0, 31]
   --Functions: 
     (1) MergeBatchnorm 
         Merge batchnorm and scale layers into the convolution layer.
         This step is necessary whenever batch normalization are used in previous training. 
     (2) MergeInputMean
         Merge the image means of the data layer into the first convolution layer
         This step is necessary whenever image means of the data layer are non-zeros. 
     (3) EditGain
         Transform the activation output of each layer to be within the range of [0, 31]
         This step is required whenever the maximal activation output of each layer is not 31
     (4) RefreshModel 
         Refresh the .prototxt and .caffemodel

"""

import sys
import os
import sys
import numpy as np
import math
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format
from argparse import ArgumentParser
from Utils import *


class RefineNetwork():

    def __init__(self, prototxt, model, input_scale, image_means, act_bits=5, last_layer_10bit=0, dst_prototxt=None, dst_model=None, log_file=None):
      
        self.prototxt = prototxt 
        self.model = model 
        self.input_scale = input_scale
        self.image_means = image_means
        self.act_bits = act_bits

        caffe.set_mode_gpu() 
        self.net = caffe.Net(prototxt, model, caffe.TEST)
        self.log_file = log_file
        self.last_layer_10bit = last_layer_10bit

        self.dst_prototxt = dst_prototxt 
        self.dst_model = dst_model
        if dst_prototxt is None:
           self.dst_prototxt = os.path.splitext(self.prototxt)[0] + '_refined_new.prototxt'
        if dst_model is None:
           self.dst_model = os.path.splitext(self.model)[0] + '_refined_new.caffemodel'


    def MergeBatchnorm(self):                                                                                                 

        if self.log_file:
           with open(self.log_file, "a+") as f:
             f.write('Merging bath normalzaition' + '\n\n')
           f.close()
        else:
           print('====> Merging batch normalization... ')                                                                                             

        temp_prototxt =  self.dst_prototxt + '.temp.pt'
        net = load_and_fill_biases(self.prototxt, self.model, temp_prototxt, None)
        net = merge_batchnorms_in_net(net)
        process_model(net, temp_prototxt, self.dst_prototxt, [pick_empty_layers], [remove_empty_layers])   

        self.net = net
        self.prototxt = self.dst_prototxt 
        self.model = self.dst_model 
        os.remove(temp_prototxt)

        self.net.save("test1-new.caffemodel")
        return self.net

    def MergeInputMean(self):

        for idx, layer in enumerate(self.net.layers):
            if layer.type == 'Convolution' or layer.type == 'QuantConvolution':
               first_conv_layer = self.net._layer_names[idx]
               break

        coef = self.net.params[first_conv_layer][0].data
        bias = np.zeros(coef.shape[0], dtype = float)

        if coef.shape[1] == len(self.image_means):
           if self.log_file:
              with open(self.log_file, "a+") as f:
                f.write('Merging input image mean into the first conv: '  +  str(first_conv_layer) + '\n\n')
              f.close()
           else:
              print('====> Merging input image mean into the first conv: %s ...'%first_conv_layer)                                                                                             

           for output_channel in range(coef.shape[0]):
               for input_channel in range(coef.shape[1]):
                   coef3x3 = coef[output_channel][input_channel][...]
                   var = np.sum(coef3x3)
                   bias[output_channel] = bias[output_channel] - var * self.image_means[input_channel]

           self.net.params[first_conv_layer][1].data[...] += self.input_scale * bias[...]                                                            

           self.net.save("test2-new.caffemodel")
           return self.net 

        else:
           if self.log_file:
              with open(self.log_file, "a+") as f:
                f.write('Warning: The number of input mean values do not equal the number of input channels!' + '\n')
                f.write('Warning: No mean merge performed' + '\n\n')
              f.close()
           else:
              print('Warning: The number of input mean values do not equal the number of input channels!')
              print('Warning: No mean merge performed!') 

           self.net.save("test2-new.caffemodel")
           return self.net 


    def EditGain(self):

        if self.log_file:
           with open(self.log_file, "a+") as f:
             f.write('Equalizing ReLU activation output ...' + '\n\n')
           f.close()
        else:
           print('====> Equalizing ReLU activation output ... ')
        
        relu_max = float(np.power(2, self.act_bits) - 1)
        conv_type = "QuantConvolution"
        relu_type = "QuantReLU"
        conv_layers = get_layer_list(self.net, conv_type)
        relu_layers = get_layer_list(self.net, relu_type)

        prev_gain = 31.0 * self.input_scale
        alpha_list = []
        gainW_list = []
        gainB_list = []

        maxW_list = [] 
        maxB_list = []
        for i in range(len(conv_layers)):
            curr_gain = np.array(self.net.params[relu_layers[i]][0].data[...])
            gain_weight =  prev_gain / curr_gain

            if self.act_bits == 10 and i == 0:
               gain_weight = (float(relu_max)/31.0)* gain_weight # only for 10-bit 

            gain_bias = relu_max / curr_gain

            alpha_list.append("%.3f"%float(curr_gain))
            gainW_list.append("%.3f"%float(gain_weight))
            gainB_list.append("%.3f"%float(gain_bias))

            prev_gain = curr_gain
            self.net.params[relu_layers[i]][0].data[...] = relu_max
            if i==len(relu_layers)-1 and self.act_bits ==5 and self.last_layer_10bit ==1:
                self.net.params[relu_layers[i]][0].data[...] = 31.96875


            weight_data = np.array(self.net.params[conv_layers[i]][0].data)
            bias_data = np.array(self.net.params[conv_layers[i]][1].data)

            self.net.params[conv_layers[i]][0].data[...] = np.array(gain_weight * weight_data)
            self.net.params[conv_layers[i]][1].data[...] = np.array(gain_bias * bias_data)
            maxW_list.append("%.3f"%float(np.amax(self.net.params[conv_layers[i]][0].data[...])))
            maxB_list.append("%.3f"%float(np.amax(self.net.params[conv_layers[i]][1].data[...])))

        if self.log_file:
           with open(self.log_file, "a+") as f:
               f.write('Layers' + '\t\t' + 'ReLU Alpha' +'\t' + 'Weight Gain' + '\t' + 
                       'Bias Gain' + '\t' + 'max weight' + '\t' + 'max bias' + '\n')
               for x in range(0, len(conv_layers)):
                   f.write(conv_layers[x] + '\t\t' + str(alpha_list[x]) + '\t\t' + str(gainW_list[x]) + 
                           '\t\t' + str(gainB_list[x]) + '\t\t' + str(maxW_list[x]) + '\t\t' + str(maxB_list[x]) + '\n')  
               f.write("scale factor:   " + str(float(alpha_list[len(conv_layers) - 1]) / relu_max) + '\n\n')

        self.net.save("test3-new.caffemodel")
        return self.net


    def RefreshModel(self):

        if self.log_file:
           with open(self.log_file, "a+") as f:
             f.write('Refreshing network prototxt and caffemodel ...' + '\n\n')
           f.close()
        else:
           print('====> Refreshing network prototxt and caffemodel ... ')
        
        relu_type = "QuantReLU"
        relu_layers = get_layer_list(self.net, relu_type)
        
        with open(self.prototxt) as f:
            net_model = caffe.proto.caffe_pb2.NetParameter()
            pb.text_format.Merge(f.read(), net_model)

            relu_max = float(np.power(2, self.act_bits) - 1)
   
            relu_count = 0    
            conv_count = 0
            for i, layer in enumerate(net_model.layer):           

                if layer.type == 'QuantConvolution':
                   layer.quant_convolution_param.shift_enable = True 
                   conv_count += 1

                if layer.type == 'QuantReLU':
                   layer.param[0].lr_mult = 0.0 
                   layer.quant_relu_param.filler.value = relu_max 
                   layer.quant_relu_param.quant_enable = True
                   if layer.name==relu_layers[-1] and self.act_bits ==5 and self.last_layer_10bit ==1:
                       layer.quant_relu_param.filler.value = 31.96875
                       layer.quant_relu_param.act_bits = 10
                   print(i,'layer name: {}; act_bits: {}; act_max: {}'.format(layer.name,layer.quant_relu_param.act_bits,layer.quant_relu_param.filler.value))
                   relu_count += 1   

            assert relu_count==conv_count, "QuantConvolution layer number must match QuantReLU layer number!"

        with open(self.dst_prototxt, 'w') as f:
           f.write(pb.text_format.MessageToString(net_model))
        
        self.net.save(self.dst_model)  


if __name__ == '__main__':

    prototxt = '/home/zhangwanchun/caffe-ssd/step4/VGG_SSD_224_5801_train_quant_org_QuantReLU.prototxt'             # The network prototxt after QuantReLU training
    model = '/home/zhangwanchun/caffe-ssd/step4/pretrain_iter_190000.caffemodel'              # The weights caffemodel after QuantReLU training
    input_scale = 1.0                         # The input scale applied to input data layer
    image_means = [0, 0, 0]             # The mean values applied to input data layer
    act_bits = 5                              # Activation bits, either 5 (5-bit) or 10 (10-bit).
    last_layer_10bit = 0                      # Last layer activation bits setting
    prototxt_refined = 'net_refined.prototxt' # prototxt after network refinement, user to fill
    model_refined = 'net_refined.caffemodel'  # caffemodel after network refinement, user to fill

    net = RefineNetwork(prototxt, model, input_scale, image_means, act_bits, last_layer_10bit, prototxt_refined, model_refined, log_file='RefineNet.log')
    #net.MergeBatchnorm()                       # Uncomment it if Batchnorm used in previous training step
    #net.MergeInputMean()                       # Uncomment it if input image mean is nonzero
    net.EditGain()           
    net.RefreshModel()
    print('Done')
