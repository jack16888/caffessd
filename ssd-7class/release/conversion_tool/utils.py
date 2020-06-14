"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
import re
import os
import sys
import tempfile
import subprocess
import datetime
import numpy as np
import json

def mktemp():
    TEMP_DIR = tempfile.mkdtemp()
    os.mkdir(os.path.join(TEMP_DIR, 'common_format'))
    return TEMP_DIR

def make_call_with(log_fname):
    def call(s):
        with open(log_fname, 'a') as f:
            f.write(s + '\n')
        if subprocess.call(s, shell=True) != 0:
            print("Error occurred during processing. Check log file {} for error messages.".format(log_fname))
            sys.exit(1)
    return call

def extract_ints_by_string(json_fname, search_str):
    with open(json_fname) as f:
        s = f.read().replace('\n', '')
        # return [ int(ival) for kvfield in re.findall(search_str + r'([^],[}]*\[[^]]*]|[^],[}]*[,}])', s) for ival in re.findall('[0-9]+', kvfield) ]
        return [[int(ival) for ival in re.findall('[0-9]+', kvfield)] for kvfield in
                re.findall(search_str + r'([^],[}]*\[[^]]*]|[^],[}]*[,}])', s)]


# returns flattened array of ints
def extract_int_by_string(json_fname, search_str):
    return [a for b in extract_ints_by_string(json_fname, search_str) for a in b]
    # with open(json_fname) as f:
    #    s = f.read().replace('\n', '')
    #    return [ int(ival) for kvfield in re.findall(search_str + r'([^],[}]*\[[^]]*]|[^],[}]*[,}])', s) for ival in re.findall('[0-9]+', kvfield) ]
    # return [ int(x) for line in f for x in re.findall('[0-9]+', line) if s in line ]


def extract_bool_by_string(json_fname, search_str):
    with open(json_fname) as f:
        s = f.read().replace('\n', '')
        return [bval == 'true' for kvfield in re.findall(search_str + r'([^],[}]*\[[^]]*]|[^],[}]*[,}])', s) for bval in
                re.findall('(true|false)', kvfield)]


# returns array of dicts containing offset and shape information per layer
# pass in 2 lengths so we can verify their sizes
def extract_net_structure(json_fname, filter_len, bias_len):
    major_layer_number = extract_int_by_string(json_fname, 'Major layer number')[0]
    sublayer_counts = [len(l) for l in extract_ints_by_string(json_fname, 'scaling')]
    # sublayer_counts = extract_int_by_string(json_fname, 'sublayer number')
    print(('sublayer counts', sublayer_counts))
    sublayer_number = sum(sublayer_counts)
    filter_1x1_flags = (x for x in (extract_int_by_string(json_fname, 'oneCoef') + [0 for _ in range(sublayer_number)]))
    depth_enable = extract_bool_by_string(json_fname, 'depthEnable')
    # print(('depth enable', depth_enable))
    in_channels = extract_int_by_string(json_fname, 'input channels')
    out_channels = extract_int_by_string(json_fname, 'output channels')
    bias_offset = 0
    weights_offset = 0
    net_info = []
    # layer_names = ( x for x in LAYER_NAME_LIST )
    for i in range(major_layer_number):
        for j in range(sublayer_counts[i]):
            filter_1x1_flag = next(filter_1x1_flags)
            # to_append = { 'layer_name': next(layer_names) }
            to_append = {}
            to_append['bias_offset'] = bias_offset
            to_append['weights_offset'] = weights_offset
            to_append['bias_length'] = in_channels[
                i] if i == 1 and j == 0 and depth_enable == True and filter_1x1_flag == 0 else out_channels[i]
            bias_offset += to_append['bias_length']
            if i == 0 and j == 0:
                # argh, make sure the input channel is 3, except when it's not
                to_append['weights_shape'] = (
                out_channels[i], 3 if in_channels[i] == 3 or in_channels[i] == 16 else in_channels[i], 3, 3)
            else:
                if depth_enable[i] and filter_1x1_flag == 0:  # depthwise
                    if j == 0:
                        to_append['weights_shape'] = (in_channels[i], 1, 3, 3)
                    else:
                        to_append['weights_shape'] = (out_channels[i], 1, 3, 3)
                elif depth_enable[i]:  # 1x1 convolution
                    if j == 0 or j == 1:
                        to_append['weights_shape'] = (out_channels[i], in_channels[i], 1, 1)
                    else:
                        to_append['weights_shape'] = (out_channels[i], out_channels[i], 1, 1)
                else:
                    if j == 0:
                        to_append['weights_shape'] = (out_channels[i], in_channels[i], 3, 3)
                    else:
                        to_append['weights_shape'] = (out_channels[i], out_channels[i], 3, 3)
            to_append['weights_length'] = to_append['weights_shape'][0] * to_append['weights_shape'][1] * \
                                          to_append['weights_shape'][2] * to_append['weights_shape'][3]
            # TODO(bowei): use these
            to_append['in_color'] = to_append['weights_shape'][1]
            to_append['out_color'] = to_append['weights_shape'][0]
            weights_offset += to_append['weights_length']
            net_info.append(to_append)

    # verify length
    if filter_len is not None and filter_len != weights_offset or bias_len is not None and bias_len != bias_offset:
        print(("ERROR", net_info, filter_len, weights_offset, bias_len, bias_offset))
        exit(1)
    return net_info


def ReSliceLayer(net, target_layer, mask_bit, weight_bit, shift_max):

    eps = 1e-6
    bias_bit = 20

    weight_data = np.array(net.params[target_layer][0].data)
    bias_data = np.array(net.params[target_layer][1].data)

    max_value = max(np.amax(np.abs(weight_data)), eps)
    max_value_bias = max(np.amax(np.abs(bias_data)), eps)

    (outCh, inCh, kh, kw) = weight_data.shape

    if mask_bit == 1:
        if weight_bit == 12:
            bias_bit = 12

        weight_num_steps = np.power(2, weight_bit - 1) - 1
        shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
        bias_num_steps = np.power(2, bias_bit - 1) - 1
        shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

        shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
        shift_power = np.power(2, shift)

        print("Layer: %s, Weight bit: %d, 1-bit Slicing, Shift: %d" % (
            target_layer, weight_bit, shift_w))


        for i in range(outCh):
            for j in range(inCh):
                coef3x3 = weight_data[i][j][...]
                var = np.sum(abs(coef3x3)) / np.prod(coef3x3.shape)
                grid_scalar = (var + eps) / 1.0
                quant_scalar = np.floor(grid_scalar * shift_power + 0.5) / float(shift_power)

                coef3x3[coef3x3 >= 0] = quant_scalar
                coef3x3[coef3x3 < 0] = -quant_scalar
                weight_data[i][j][...] = coef3x3

            if bias_data[i] >= 0:
                bias_data[i] = np.floor(bias_data[i] * shift_power + 0.5) / float(shift_power)
            else:
                bias_data[i] = np.ceil(bias_data[i] * shift_power - 0.5) / float(shift_power)

        net.params[target_layer][0].data[...] = weight_data
        net.params[target_layer][1].data[...] = bias_data


    elif mask_bit == 2:
        if weight_bit == 12:
            bias_bit = 12

        weight_num_steps = np.power(2, weight_bit - 1) - 1
        shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
        bias_num_steps = np.power(2, bias_bit - 1) - 1
        shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

        shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
        shift_power = np.power(2, shift)
        print("Layer: %s, Weight bit: %d, 2-bit Slicing, Shift: %d" % (
            target_layer, weight_bit, shift))

        for i in range(outCh):
            for j in range(inCh):
                coef3x3 = weight_data[i][j][...]
                grid_scalar = np.sum(abs(coef3x3)) / np.prod(coef3x3.shape)
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

        net.params[target_layer][0].data[...] = weight_data
        net.params[target_layer][1].data[...] = bias_data


    elif mask_bit == 3:
        if weight_bit == 12:
            bias_bit = 18

        weight_num_steps = np.power(2, weight_bit - 1) - 1
        shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
        bias_num_steps = np.power(2, bias_bit - 1) - 1
        shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

        shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
        shift_power = np.power(2, shift)

        print("Layer: %s, Weight bit: %d, 3-bit Slicing, Shift: %d" % (
            target_layer, weight_bit, shift_w))

        for i in range(outCh):
            for j in range(inCh):
                coef3x3 = weight_data[i][j][...]
                var = np.sum(abs(coef3x3)) / np.prod(coef3x3.shape)
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

        net.params[target_layer][0].data[...] = weight_data
        net.params[target_layer][1].data[...] = bias_data

    elif mask_bit == 5:
        if weight_bit == 12:
            bias_bit = 16

        weight_num_steps = np.power(2, weight_bit - 1) - 1
        shift_w = int(np.floor(np.log2(weight_num_steps / max_value)))
        bias_num_steps = np.power(2, bias_bit - 1) - 1
        shift_b = int(np.floor(np.log2(bias_num_steps / max_value_bias)))

        shift = np.clip(np.amin([shift_w, shift_b]), 0, shift_max)
        shift_power = np.power(2, shift)

        print("Layer: %s, Weight bit: %d, 5-bit Slicing, Shift: %d" % (
             target_layer, weight_bit, shift_w))

        n_bit = mask_bit
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

        net.params[target_layer][0].data[...] = weight_data
        net.params[target_layer][1].data[...] = bias_data

    elif mask_bit >= 8:  # 8 bit or 12 bit

        weight_num_steps = np.power(2, weight_bit - 1) - 1
        shift_w = np.clip(int(np.floor(np.log2(weight_num_steps / max_value))), 0, shift_max)
        shift_power = np.power(2, shift_w)

        print("Layer: %s, %d-bit Slicing, Shift: %d" % (
             target_layer, mask_bit, shift_w))

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

        net.params[target_layer][0].data[...] = weight_data
        net.params[target_layer][1].data[...] = bias_data

    return net

