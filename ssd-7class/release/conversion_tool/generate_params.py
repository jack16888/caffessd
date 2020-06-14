"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
from __future__ import print_function
import os
import sys
import caffe
import numpy as np
from argparse import ArgumentParser
from gen_txts import *
from utils import extract_ints_by_string, extract_int_by_string
import json


def main(args):
    caffe.set_mode_cpu()
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    CONV_TYPE_NAME = 'QuantConvolution'
    RELU_TYPE_NAME = 'QuantReLU'
    INPUT_SCALE = float(args.input_scale)
    sublayers = extract_int_by_string(args.template_json, 'sublayer_number')
    NUM_LAYERS = sum(sublayers)
    coef_bits = extract_int_by_string(args.template_json, 'coef_bits')
    coef_bits_list = []
    for i, layer_num in enumerate(sublayers):
        for j in range(layer_num):
            coef_bits_list.append(coef_bits[i])
    conv_layer_names = GetLayerList(net, NUM_LAYERS, CONV_TYPE_NAME)
    relu_layer_names = GetLayerList(net, NUM_LAYERS, RELU_TYPE_NAME)

    net, shift_list = ReSliceModel(net, coef_bits_list, conv_layer_names, int(args.weight_bits), int(args.shift_max))

    if args.edit_gain == 'True' or args.edit_gain == 'true':
        net = EditGain(net, INPUT_SCALE, conv_layer_names, relu_layer_names, args.log_outfname)
        shift_list = CalculateShift(net, int(args.weight_bits), conv_layer_names, int(args.shift_max), args.log_outfname)
    elif args.edit_gain == 'False' or args.edit_gain == 'false':
        pass
    else:
        print('Improper value for edit gain parameter, expected [Tt]rue/[Ff]alse, got ' + args.edit_gain)
        exit(1)

    net_config = UpdateNetJson(args.template_json, shift_list, args.json_outfname, args.evaluate_path)

    CompactWeights(net, conv_layer_names, args.filter_outfname, args.bias_outfname)

    if args.debug == 'True' or args.edit_gain == 'true':
        GenerateVChipParams(net, shift_list, conv_layer_names, args.temp_dir)

    with open(args.fullmodel_def_json) as jf:
        jj = json.load(jf)
        fc_kwargs = {o['name']: os.path.join(os.path.dirname(args.fc_outfname), o['data file']) for o in jj['layer'] if
                     o['operation'] == 'FC'}
        GenerateFCParams(net, **fc_kwargs)

def gen_txts_parse_args(argv):
    parser = ArgumentParser(description="generate txt files for conversion")
    parser.add_argument('model', help='the net definition prototxt')
    parser.add_argument('weights', help='the weights caffemodel')
    parser.add_argument('template_json', help='the template net json')
    parser.add_argument('fullmodel_def_json', help='the template full model def json. Contains info about fc layers')
    parser.add_argument('weight_bits', help='weight_bits')
    parser.add_argument('shift_max', help='shift_max')
    parser.add_argument('input_scale', help='input_scale')
    parser.add_argument('edit_gain', help='whether or not to edit gain? Default should be false')
    parser.add_argument('debug', help='whether or not use deubg mode? Default should be false')
    parser.add_argument('evaluate_path', help='the path to evaluate image or images directory')
    parser.add_argument('\'-o\'', help='should be the string -o')
    parser.add_argument('temp_dir', help='temporary output directory')
    parser.add_argument('filter_outfname', help='output filter.txt location')
    parser.add_argument('bias_outfname', help='output bias.txt location')
    parser.add_argument('json_outfname', help='output net.json location (scaling shifts inserted)')
    parser.add_argument('fc_outfname', help='output fc.bin location')
    parser.add_argument('log_outfname', help='output text log location, optional', nargs='?', default=None)
    args = parser.parse_args(['_' if x == '-o' else x for x in argv[1:]])
    return args

if __name__ == '__main__':
    main(gen_txts_parse_args(sys.argv))
