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
************************************************************************

THIRD-PARTY SOFTWARE NOTICES AND INFORMATION

This software, incorporates material from the project(s)
listed below (collectively, "Third Party Code").

1. Faster R-CNN

The MIT License (MIT)

Copyright (c) 2015 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

************END OF THIRD-PARTY SOFTWARE NOTICES AND INFORMATION**********
"""

import numpy as np
import sys
import os
import os.path as osp
import google.protobuf as pb
from argparse import ArgumentParser
import caffe
import google.protobuf.text_format

def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution' or layer.type == 'QuantConvolution':
            # Add bias layer if needed
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0


    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)
    net_dst = caffe.Net(dst_model, caffe.TEST)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[...] = net_src.params[key][i].data[...]

    if dst_weights is not None:
        # Store params
        pass

    return net_dst


def merge_conv_and_bn(net, i_conv, i_bn, i_scale):
    assert(i_conv != None)
    assert(i_bn != None)

    def copy_double(data):
        return np.array(data, copy=True, dtype=np.double)

    key_conv = net._layer_names[i_conv]
    key_bn = net._layer_names[i_bn]
    key_scale = net._layer_names[i_scale] if i_scale else None

    # Copy
    bn_mean = copy_double(net.params[key_bn][0].data)
    bn_variance = copy_double(net.params[key_bn][1].data)
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    #if net.params.has_key(key_scale):
    if key_scale in net.params:
        print('Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale))
        scale_weight = copy_double(net.params[key_scale][0].data)
        scale_bias = copy_double(net.params[key_scale][1].data)
        net.params[key_scale][0].data[:] = 1
        net.params[key_scale][1].data[:] = 0
    else:
        print('Combine {:s} + {:s}'.format(key_conv, key_bn))
        scale_weight = 1
        scale_bias = 0

    weight = copy_double(net.params[key_conv][0].data)
    bias = copy_double(net.params[key_conv][1].data)
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + 1e-5)
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
    for i in range(len(alpha)):
        net.params[key_conv][0].data[i] = weight[i] * alpha[i]

def merge_batchnorms_in_net(net):
    # for each BN
    for i, layer in enumerate(net.layers):
        if layer.type != 'BatchNorm':
            continue

        l_name = net._layer_names[i]

        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]

        can_be_absorbed = True

        # Search all (bottom) layers
        for j in range(i - 1, -1, -1):
            tops_of_j = net.top_names[net._layer_names[j]]
            if l_bottom in tops_of_j:
                if net.layers[j].type not in ['Convolution', 'InnerProduct', 'QuantConvolution']:
                    can_be_absorbed = False
                else:
                    # There must be only one layer
                    conv_ind = j
                    break

        if not can_be_absorbed:
            continue

        # find the following Scale
        scale_ind = None
        for j in range(i + 1, len(net.layers)):
            bottoms_of_j = net.bottom_names[net._layer_names[j]]
            if l_top in bottoms_of_j:
                if scale_ind:
                    # Followed by two or more layers
                    scale_ind = None
                    break

                if net.layers[j].type in ['Scale']:
                    scale_ind = j

                    top_of_j = net.top_names[net._layer_names[j]][0]
                    if top_of_j == bottoms_of_j[0]:
                        # On-the-fly => Can be merged
                        break

                else:
                    # Followed by a layer which is not 'Scale'
                    scale_ind = None
                    break

        merge_conv_and_bn(net, conv_ind, i, scale_ind)

    return net

def process_model(net, src_model, dst_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    bn_count = 0
    for i, layer in enumerate(model.layer):
        if (layer.type == 'BatchNorm') and (bn_count == 0):
           first_bn = layer.name
           bn_count += 1

    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i, first_bn), func_loop)

    map(lambda x: x(net, model), func_finally)

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i, first_bn):
    if layer.type not in ['BatchNorm', 'Scale']:
        return

    bottom = layer.bottom[0]
    top = layer.top[0]

    if (bottom != top):
        # Not supperted yet
        return

    if layer.type == 'BatchNorm':
        zero_mean = np.all(net.params[layer.name][0].data == 0)
        one_var = np.all(net.params[layer.name][1].data == 1)
        length_is_1 = (net.params[first_bn][2].data == 1) or (net.params[layer.name][2].data == 0)

        if zero_mean and one_var and length_is_1:
            print('Delete layer: {}'.format(layer.name))
            to_delete_empty.append(layer)

    if layer.type == 'Scale':
        no_scaling = np.all(net.params[layer.name][0].data == 1)
        zero_bias = np.all(net.params[layer.name][1].data == 0)

        if no_scaling and zero_bias:
            print('Delete layer: {}'.format(layer.name))
            to_delete_empty.append(layer)

def remove_empty_layers(net, model):
    map(model.layer.remove, to_delete_empty)

# A function to add 'engine: CAFFE' param into 1x1 convolutions
def set_engine_caffe(layer, net, model, i):
    if layer.type == 'Convolution' or layer.type == 'QuantConvolution':
        if layer.convolution_param.kernel_size == 1\
            or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']

def get_layer_list(net, layer_type):
    
    layer_list = []
    layer_count = 0
    for idx, layer in enumerate(net.layers):
        layer_name = net._layer_names[idx]
        if layer.type == layer_type:
            layer_list.append(layer_name)
            layer_count += 1

    return layer_list


def get_layer_names(target_layer_names):                                                                                      
    if isinstance(target_layer_names, list):                                                                                        
        print('Get layer names from list')                                                                             
        return target_layer_names                                                                                                   
    if isinstance(target_layer_names, str):                                                                                         
        print('Read layer names from file, {}'.format(target_layer_names))                                   
        layer_names = []                                                                                                            
        with open(target_layer_names) as fp:                                                                                        
            line = fp.readline()                                                                                                    
            while line:                                                                                                             
                if line.strip() is not '':
                    layer_names.append(line.strip())                                                                                    
                line = fp.readline()                                                                                                
        return layer_names                                                                                                          

def get_input_image_means(input_means):                                                                                      
    if isinstance(input_means, list):                                                                                        
        print('Get input image means from list')                                                                             
        print('Input image means: ', input_means)                                                                             
        return input_means                                                                                                   
    if isinstance(input_means, str):                                                                                         
        print('Read input image means from file, {}\n'.format(input_means))                                   
        image_means = []                                                                                                            
        with open(input_means) as fp:                                                                                        
            line = fp.readline()                                                                                                    
            while line:                                                                                                             
                if line.strip() is not '':
                    image_means.append(line.strip())                                                                                    
                line = fp.readline()                                                                                                
        print('Input image means: {}'.format(input_means))                                                                             
        return image_means                                                                                                          
