#!/usr/bin/env python
"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
from argparse import ArgumentParser
import caffe
import json
import numpy as np
import os
import re
import shutil
import sys
import struct

def GetLayerList(net, layer_size, layer_type):
    layer_list = []
    layer_count = 0
    for idx, layer in enumerate(net.layers):
        layer_name = net._layer_names[idx]
        if layer.type == layer_type and layer_count < layer_size:
            layer_list.append(layer_name)
            layer_count += 1

    return layer_list

# Extract conv and relu layers up to the maximum supported by chip
def ExtractLayers(old_model, old_weights, conv_size, type_dict):
    old_net = caffe.Net(old_model, old_weights, caffe.TEST)

    old_conv_list = []
    conv_idx = 0
    old_relu_list = []
    relu_idx = 0

    for i, layer in enumerate(old_net.layers):
        layer_name = old_net._layer_names[i]
        if layer.type == type_dict['old_conv_type'] and conv_idx < conv_size:
            old_conv_list.append(layer_name)
            conv_idx += 1
        if layer.type == type_dict['old_relu_type'] and relu_idx < conv_size:
            old_relu_list.append(layer_name)
            relu_idx += 1

    return old_net, old_conv_list, old_relu_list

# edit the net params in-place by merging gain
def EditGain(net, input_scale, conv_layers, relu_layers, log_outfname):
    prev_gain = 255 * input_scale
    active_max = 31
    alpha_list = []

    for i in range(len(conv_layers)):
        print(conv_layers[i], ", ", relu_layers[i])

        curr_gain = np.array(net.params[relu_layers[i]][0].data[...])
        print("prev: ", prev_gain, "curr: ", curr_gain)
        alpha_list.append(float(curr_gain))

        gain_weight = prev_gain / curr_gain
        gain_bias = active_max / curr_gain

        prev_gain = curr_gain
        net.params[relu_layers[i]][0].data[...] = active_max

        weight_data = np.array(net.params[conv_layers[i]][0].data)
        bias_data = np.array(net.params[conv_layers[i]][1].data)

        net.params[conv_layers[i]][0].data[...] = np.array(gain_weight * weight_data)
        net.params[conv_layers[i]][1].data[...] = np.array(gain_bias * bias_data)

    # TODO: handle bias for the layer after the conv layer

    if log_outfname:
        with open(log_outfname, "w+") as f:
            f.write("relu alphas: \n")
            for x in range(0, len(conv_layers)):
                f.write('\t' + conv_layers[x] + ": " + str(alpha_list[x]) + '\n')
            f.write("scale factor: " + str(alpha_list[len(conv_layers) - 1] / active_max) + '\n')
    return net

def CalculateShift(net, weight_bits, conv_layers, shift_max, log_outfname):
    weight_num_steps = np.power(2, weight_bits - 1) - 1
    bias_bits = 20
    offset_ratio = np.power(2, bias_bits - weight_bits)

    shift_list = []

    with open(log_outfname, "a+") if log_outfname else sys.stderr as log:
        if log_outfname:
            log.write("bias offset ratio: " + str(offset_ratio) + '\n')
        for layer in conv_layers:
            weight_max = np.amax(net.params[layer][0].data[...])
            weight_min = np.amin(net.params[layer][0].data[...])

            bias_max = np.amax(net.params[layer][1].data[...])
            bias_min = np.amin(net.params[layer][1].data[...])

            max_value = max(weight_max, - weight_min, bias_max / offset_ratio, -bias_min / offset_ratio)

            shift = int(np.floor(np.log2(weight_num_steps / max_value)))
            if shift > shift_max:
                shift = shift_max
            shift_list.append(shift)
            print("max_value: ", max_value, " shift_value: ", shift)
            if log_outfname:
                log.write(layer + ": \n")
                log.write("\tmax_value: " + str(max_value) + " shift value: " + str(shift) + '\n')
                log.write("\tweight_max: " + str(weight_max) + " weight_min: " + str(weight_min) + " bias_max: " \
                          + str(bias_max) + " bias_min: " + str(bias_min) + '\n')
    return shift_list

# shift_list needs to be array of ints
def UpdateNetJson(template_json_file, shift_list, json_outfname, evaluate_path):
    with open(template_json_file) as f:
        net_config = json.load(f)

    # add MajorLayerNumber
    net_config['model'][0]['MajorLayerNumber'] = len(net_config['layer'])
    
    # add major_layer and shift values to net.json
    idx = 0
    for i, layer in enumerate(net_config['layer']):
        layer['major_layer'] = i + 1
        layer['scaling'] = []
        for i in range(layer['sublayer_number']):
            layer['scaling'].append(int(shift_list[idx]))
            idx += 1
    
    #change net.json learning mode to do the conversion
    if evaluate_path != 'None':
        if os.path.isdir(evaluate_path): # need turn off all the learning mode
            for layer in net_config['layer']:
                if 'learning' in layer and layer['learning']:
                    layer['learning'] = False          
        elif os.path.isfile(evaluate_path):  # need turn on all the learning mode
            for layer in net_config['layer']:
                if 'learning' not in layer or not layer['learning']:
                    layer['learning'] = True

    with open(json_outfname, 'w') as f:
        json.dump(net_config, f, indent=4, separators=(',', ': '), sort_keys=True)
    return net_config   

# dumps CNN parameters into txt files
def CompactWeights(net, conv_layers, filter_path, bias_path):
    flt = np.array([])
    bias = np.array([])
    for layer in conv_layers:
        flt = np.concatenate((flt, np.array(net.params[layer][0].data[...]).flatten()), axis=0)
        bias = np.concatenate((bias, np.array(net.params[layer][1].data[...]).flatten()), axis=0)
    np.savetxt(filter_path, flt, fmt='%.16e', delimiter='\n')
    np.savetxt(bias_path, bias, fmt='%.16e', delimiter='\n')


# dump vchip cnn.param and fc.param for debug
def GenerateVChipParams(net, shift_list, conv_layer, output_dir):
    conv_param_file = os.path.join(output_dir, "cnn.param")
    fc_param_file = os.path.join(output_dir, "fc.param")
    shift_param_file = os.path.join(output_dir, "fxp_shift_bits.txt")

    shift = np.array(shift_list).astype(np.int32)
    with open(conv_param_file, 'wb') as conv_file:
        shift.tofile(conv_file)
        for idx, layer in enumerate(conv_layer):
            # weights/bias
            conv_w = net.params[layer][0].data.flatten() * (1 << shift[idx])
            conv_w_fix = conv_w.round().astype(np.int32)
            conv_b = net.params[layer][1].data.flatten() * (1 << shift[idx])
            conv_b_fix = conv_b.round().astype(np.int32)

            # write to file
            conv_file.write(conv_w_fix)
            conv_file.write(conv_b_fix)

    with open(fc_param_file, 'wb') as fc_file:
        fclayers = []
        for layer in net._layer_names:
            if 'fc' in layer:
                fclayers.append(layer)
        for fcname in fclayers:
            fc_file.write(net.params[fcname][0].data)
            fc_file.write(net.params[fcname][1].data)

    with open(shift_param_file, 'wb') as shift_file:
        for shift in shift_list:
            shift_file.write("%d\n" % shift)

def ReSliceModel(net, mask_bit, conv_layers, weight_bit, shift_max):
    layer_count = 0

    shift_list = []
    eps = 1e-6

    bias_bit = 20
    for layer in conv_layers:

        weight_data = np.array(net.params[layer][0].data)
        bias_data = np.array(net.params[layer][1].data)

        max_value = max(np.amax(np.abs(weight_data)), eps)
        max_value_bias = max(np.amax(np.abs(bias_data)), eps)

        (outCh, inCh, kh, kw) = weight_data.shape

        if mask_bit[layer_count] == 1:
            if weight_bit == 12:
                bias_bit = 12

            weight_num_steps = np.power(2, weight_bit - 1) - 1
            shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
            bias_num_steps = np.power(2, bias_bit - 1) - 1
            shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

            shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
            shift_power = np.power(2, shift)
            shift_list.append(shift)

            print("Layer count: %d, Layer: %s, Weight bit: %d, 1-bit Slicing, Shift: %d" % (
                layer_count, layer, weight_bit, shift_w))


            for i in range(outCh):
                for j in range(inCh):
                    coef3x3 = weight_data[i][j][...]
                    var = sum(abs(coef3x3).flatten()) / np.prod(coef3x3.shape)
                    grid_scalar = (var + eps) / 1.0
                    quant_scalar = np.floor(grid_scalar * shift_power + 0.5) / float(shift_power)

                    coef3x3[coef3x3 >= 0] = quant_scalar
                    coef3x3[coef3x3 < 0] = -quant_scalar
                    weight_data[i][j][...] = coef3x3

                if bias_data[i] >= 0:
                    bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
                else:
                    bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

            net.params[layer][0].data[...] = weight_data
            net.params[layer][1].data[...] = bias_data


        elif mask_bit[layer_count] == 2:
            if weight_bit == 12:
                bias_bit = 12

            weight_num_steps = np.power(2, weight_bit - 1) - 1
            shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
            bias_num_steps = np.power(2, bias_bit - 1) - 1
            shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

            shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
            shift_power = np.power(2, shift)
            shift_list.append(shift)
            print("Layer count: %d, Layer: %s, Weight bit: %d, 2-bit Slicing, Shift: %d" % (
                layer_count, layer, weight_bit, shift))

            for i in range(outCh):
                for j in range(inCh):
                    coef3x3 = weight_data[i][j][...]
                    grid_scalar = sum(abs(coef3x3).flatten()) / np.prod(coef3x3.shape)
                    quant_scalar = np.floor(grid_scalar * shift_power + 0.5) / float(shift_power)

                    for m in range(kh):
                        for n in range(kw):
                            if np.abs(coef3x3[m,n]) < 0.25*quant_scalar:
                                coef3x3[m, n] = 0.0
                            elif coef3x3[m,n] >= 0: 
                                coef3x3[m, n] = quant_scalar
                            else:
                                coef3x3[m, n] = -quant_scalar
                    weight_data[i][j][...] = coef3x3

                if bias_data[i] >= 0:
                    bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
                else:
                    bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

            net.params[layer][0].data[...] = weight_data
            net.params[layer][1].data[...] = bias_data


        elif mask_bit[layer_count] == 3:
            if weight_bit == 12:
                bias_bit = 18

            weight_num_steps = np.power(2, weight_bit - 1) - 1
            shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
            bias_num_steps = np.power(2, bias_bit - 1) - 1
            shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

            shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
            shift_power = np.power(2, shift)
            shift_list.append(shift)

            print("Layer count: %d, Layer: %s, Weight bit: %d, 3-bit Slicing, Shift: %d" % (
                layer_count, layer, weight_bit, shift_w))

            for i in range(outCh):
                for j in range(inCh):
                    coef3x3 = weight_data[i][j][...]
                    var = sum(abs(coef3x3).flatten()) / np.prod(coef3x3.shape)
                    grid_scalar = (var + eps) / 4.0
                    quant_scalar = np.floor(grid_scalar * shift_power + 0.5) / float(shift_power)

                    for m in range(kh):
                        for n in range(kw):
                            abs_coefInt = int(abs(coef3x3[m, n]) / grid_scalar)
                            if abs_coefInt > 2:
                                abs_coefInt = 4
                            abs_coef = abs_coefInt * quant_scalar

                            if coef3x3[m, n] >= 0:
                                coef3x3[m, n] = abs_coef
                            else:
                                coef3x3[m, n] = -abs_coef

                    weight_data[i][j][...] = coef3x3

                if bias_data[i] >= 0:
                    bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
                else:
                    bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

            net.params[layer][0].data[...] = weight_data
            net.params[layer][1].data[...] = bias_data

        elif mask_bit[layer_count] == 5:
            if weight_bit == 12:
                bias_bit = 16

            weight_num_steps = np.power(2, weight_bit - 1) - 1
            shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
            bias_num_steps = np.power(2, bias_bit - 1) - 1
            shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

            shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
            shift_power = np.power(2, shift)
            shift_list.append(shift)

            print("Layer count: %d, Layer: %s, Weight bit: %d, 5-bit Slicing, Shift: %d" % (
                layer_count, layer, weight_bit, shift_w))

            n_bit = mask_bit[layer_count]
            mask_num_steps = np.power(2, n_bit-1) - 1
            for i in range(outCh):
                for j in range(inCh):
                    coef3x3 = weight_data[i][j][...]
                    var = np.amax(abs(coef3x3))
                    grid_scalar = (var + eps) / float(mask_num_steps)
                    quant_scalar = np.floor(grid_scalar * shift_power + 0.5) / float(shift_power)

                    for m in range(kh):
                        for n in range(kw):
                            if coef3x3[m, n] >= 0:
                                coef3x3[m, n] = np.floor(coef3x3[m, n] / grid_scalar + 0.5) * quant_scalar
                            else:
                                coef3x3[m, n] = np.ceil(coef3x3[m, n] / grid_scalar - 0.5) * quant_scalar

                    weight_data[i][j][...] = coef3x3

                if bias_data[i] >= 0:
                    bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
                else:
                    bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

            net.params[layer][0].data[...] = weight_data
            net.params[layer][1].data[...] = bias_data

        elif mask_bit[layer_count] >= 8:  # 8 bit or 12 bit

            weight_num_steps = np.power(2, weight_bit - 1) - 1
            shift_w = np.clip(int(np.floor(np.log2(weight_num_steps / max_value))), 0, shift_max)
            shift_power = np.power(2, shift_w)
            shift_list.append(shift_w)

            print("Layer count: %d, Layer: %s, %d-bit Slicing, Shift: %d" % (
                layer_count, layer, mask_bit[layer_count], shift_w))

            for i in range(outCh):
                for j in range(inCh):
                    for m in range(kh):
                        for n in range(kw):
                            if weight_data[i][j][m][n] > 0:
                                weight_data[i][j][m][n] = np.floor(
                                    (weight_data[i][j][m][n] * shift_power + 0.5)) / float(shift_power)
                            else:
                                weight_data[i][j][m][n] = np.ceil(
                                    (weight_data[i][j][m][n] * shift_power - 0.5)) / float(shift_power)

                if bias_data[i] >= 0:
                    bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
                else:
                    bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

            net.params[layer][0].data[...] = weight_data
            net.params[layer][1].data[...] = bias_data

        layer_count += 1
    return net, shift_list

# net: a Caffe .net instance with fc layers named like 'fc*'
# output_fc_Coef: name of file , eg 'fc.bin'
# kwargs: more fc file names, eg fc6='fc6.bin',fc7='fc7.bin'
def GenerateFCParams(net, output_fc_Coef=None, **kwargs):
    out_fnames = {'fc': output_fc_Coef} if output_fc_Coef is not None else kwargs
    # create output file
    fclayers = {k: [] for k in out_fnames}  # fc -> all the fc layers, fc6 -> only that layer
    for layer in net._layer_names:
        for lookup_key in fclayers:
            if layer[:len(lookup_key)] == lookup_key:
                fclayers[lookup_key].append(layer)

    for fc_name_key in fclayers:
        with open(out_fnames[fc_name_key], 'wb') as fpout:
            # write headers
            for fcname in fclayers[fc_name_key]:
                inlen = net.params[fcname][0].data.shape[1]
                outlen = net.params[fcname][1].data.shape[0]
                fpout.write(struct.pack('<i', inlen))
                fpout.write(struct.pack('<i', outlen))
                # print("Layer", fcname, 'in:', inlen, 'out:', outlen)

            for fcname in fclayers[fc_name_key]:
                fpout.write(net.params[fcname][0].data)
                fpout.write(net.params[fcname][1].data)

def GetLabelPath(fullmodel_def_json):
    label_file = ""
    with open(fullmodel_def_json) as jf:
        jj = json.load(jf)
        for layer in jj['layer']:
            if layer['operation'] == 'LABEL':
                label_file = layer['data file']
    return label_file

