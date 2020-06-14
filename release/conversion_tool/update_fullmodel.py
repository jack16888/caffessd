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

class FullModelConfig(object):
    def __init__(self, fullmodel_config):
        self.fullmodel_config = fullmodel_config
        if "version" not in self.fullmodel_config:
            self.fullmodel_config['version'] = 100
        
        self.net_confg = None
        self.layer_idx = 0
        self.chip_type = 0

    def UpdateFullModel(self, layer_idx, net_config):
        self.net_config = net_config
        self.layer_idx = layer_idx

        cnn_layer = self.fullmodel_config['layer'][self.layer_idx]
        #add inputs array to cnn layer
        cnn_layer['inputs'] = [{
            "format": "byte",
            "prefilter": "interlace_tile_encode",
            "shape": [
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['image_size'],
                self.net_config['layer'][0]['input_channels']
            ]
        }]
        #add outputs array to cnn layer, consider learning mode, the implementation vary by chip type
        cnn_layer['outputs'] = []
        DEFAULT_TILE_SIZE = 14
        NUM_ENGINES = 16

        for idx, layer in enumerate(self.net_config['layer']):
            image_size = layer['image_size']
            output_channels = layer['output_channels']
            layer_scaledown = 0
            upsample_mode = 0
            if 'upsample_enable' in layer and layer['upsample_enable']:
                image_size <<= 1
                output_channels = ((NUM_ENGINES - 1 + output_channels) / NUM_ENGINES) * NUM_ENGINES
                upsample_mode = 1
            output_format = 'byte'
            filter_type = "interlace_tile_decode" 
            if 'ten_bits_enable' in layer and layer['ten_bits_enable']:
                output_format = 'float'
                filter_type = 'interlace_tile_10bits_decode'
            tile_size = image_size if image_size < DEFAULT_TILE_SIZE else DEFAULT_TILE_SIZE
            output_size = image_size * image_size * output_channels * 32 / 49
            if 'learning' in layer and layer['learning']:
                # check fake layer 
                sublayers = layer['sublayer_number'] + 1 if self.NeedFakeLayer(layer) else layer['sublayer_number']
                for i in range(sublayers):
                    sub_output_channels = output_channels
                    #handle mobilenet one by one convolution 
                    if i == 0 and self.DepthEnabled(layer):
                        sub_output_channels = layer['input_channels']
                        sub_output_size = image_size * image_size * sub_output_channels * 32 / 49
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                sub_output_channels,
                                tile_size * tile_size,
                                sub_output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
                    else:
                        cnn_layer["outputs"].append({
                            "format": output_format,
                            "postfilter": filter_type,
                            "shape": [
                                image_size,
                                image_size,
                                output_channels,
                                tile_size * tile_size,
                                output_size
                            ],
                            "layer scaledown": layer_scaledown,
                            "upsampling": upsample_mode
                        })
            elif 'last_layer_out' in layer and layer['last_layer_out']:
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
            elif idx + 1 == len(self.net_config['layer']): # add the last layer output
                if 'pooling' in layer and layer['pooling']:
                    image_size >>= 1
                    tile_size = DEFAULT_TILE_SIZE >> 1
                    if image_size == 7: #fc_mode
                        filter_type = "fc77_decode"
        
                if filter_type == "fc77_decode":
                    output_size = image_size * image_size * output_channels * 64 / 49
                    layer_scaledown = 3
                else:
                    output_size = image_size * image_size * output_channels * 32 / 49
                cnn_layer["outputs"].append({
                    "format": output_format,
                    "postfilter": filter_type,
                    "shape": [
                        image_size,
                        image_size,
                        output_channels,
                        tile_size * tile_size,
                        output_size
                    ],
                    "layer scaledown": layer_scaledown,
                    "upsampling": upsample_mode
                })
        return self.fullmodel_config

    def NeedFakeLayer(self, layer):
        return 'resnet_shortcut_start_layers' in layer and 'pooling' in layer and layer['pooling'] and \
            layer['sublayer_number'] == layer['resnet_shortcut_start_layers'][-1] + 1

    def DepthEnabled(self, layer):
        return 'depth_enable' in layer and layer['depth_enable'] \
            and 'one_coef' in layer and len(layer['one_coef']) > 0 \
            and layer['one_coef'][0] == 0

def update_fullmodel(fullmodel_infile, work_dir, evaluate=False):
    net_infile = os.path.join(work_dir, 'net.json')
    fullmodel_outfile = os.path.join(work_dir, 'fullmodel.json')
    
    with open(fullmodel_infile) as modeljf:
        fullmodel_config = json.load(modeljf)    
    if len(fullmodel_config['layer']) < 2 or fullmodel_config['layer'][1]['name'] != 'cnn':
        sys.exit("fullmodel*.json format error!")

    with open(net_infile) as f:
        net_config = json.load(f)

    chip_nums = 1 if 'ChipNumber' not in net_config['model'][0] else net_config['model'][0]['ChipNumber']

    fullmodel = FullModelConfig(fullmodel_config)

    if chip_nums == 1:
        fullmodel_config = fullmodel.UpdateFullModel(1, net_config)
    elif chip_nums > 1:
        for i in range(chip_nums):
            with open(os.path.join(work_dir, "net_chip" + str(i) + ".json")) as subf:
                sub_net_config = json.load(subf)
            fullmodel_config = fullmodel.UpdateFullModel(i+1, sub_net_config)

    # keep DATA and CNN layers for bit match
    if evaluate:
        fullmodel_config['layer'] = fullmodel_config['layer'][:(chip_nums+1)]

    with open(fullmodel_outfile, "w") as f:
        json.dump(fullmodel_config, f, indent=4, separators=(',', ': '), sort_keys=True)    
