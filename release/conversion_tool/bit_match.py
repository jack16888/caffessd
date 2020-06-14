"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
from __future__ import print_function
import sys
import os
import numpy as np
import caffe
import cv2
import gtilib
import json
import filecmp
import shutil
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
from argparse import ArgumentParser
from utils import make_call_with

class BitMatch():
    def __init__(self, args):
        caffe.set_mode_gpu()
        self.caffe_prototxt = args.caffe_prototxt
        self.caffe_model = args.caffe_model
        self.net_json = args.net_json
        self.gti_model = args.gti_model
        self.evaluate_path = args.evaluate_path
        self.output_dir = args.output_dir
        self.log_fname = args.log_fname
        self.img_bin = os.path.join(self.output_dir, "image.bin")

        self.caffe_net = caffe.Net(self.caffe_prototxt, self.caffe_model, caffe.TEST)
        with open(self.caffe_prototxt) as f:
            self.caffe_net_txt = caffe_pb2.NetParameter()
            text_format.Merge(f.read(), self.caffe_net_txt)

        with open(self.net_json) as j:
            self.net_config = json.load(j)

        self.GTI_IMAGE_WIDTH = self.GTI_IMAGE_HEIGHT = self.net_config['layer'][0]['image_size']
        self.INPUT_CHANNELS = self.net_config['layer'][0]['input_channels']

        self.caffe_output_dir = os.path.join(self.output_dir, "caffe")
        try:
            os.makedirs(self.caffe_output_dir)
        except OSError as e:
            pass
        
        self.chip_output_dir = os.path.join(self.output_dir, "chip")
        try:
            os.makedirs(self.chip_output_dir)
        except OSError as e:
            pass

        self.OUTPUT_CHANNELS = self.net_config['layer'][-1]['output_channels']
        self.OUTPUT_IMAGE_SIZE = self.net_config['layer'][-1]['image_size']
        if 'pooling' in self.net_config['layer'][-1] and self.net_config['layer'][-1]['pooling']:
            self.OUTPUT_IMAGE_SIZE >>= 1
        if 'upsample_enable' in self.net_config['layer'][-1] and self.net_config['layer'][-1]['upsample_enable']:
            self.OUTPUT_IMAGE_SIZE <<= 1

        self.caffe_layers = []
        self.chip_layers = []
        self.GTIMODEL = ""

    def get_caffe_layers(self):
        caffe_layers = []
        for idx, layer in enumerate(self.caffe_net.layers):
            if idx < len(self.caffe_net.layers) - 1:
                layer_name = self.caffe_net._layer_names[idx]
                layer_name_next = self.caffe_net._layer_names[idx + 1]
                if layer.type == "QuantConvolution":
                    caffe_layers.append(layer_name_next) if self.caffe_net.layers[idx+1].type == "Eltwise" else caffe_layers.append(layer_name)
        return caffe_layers

    def get_chip_layers(self):
        chip_layers = []
        prefix = "dump_sublayer"
        for layer in self.net_config['layer']:
            major_layer = str(layer['major_layer'])
            for i in range(layer['sublayer_number']):
                sub_layer = str(i+1)
                chip_layers.append(prefix + major_layer + '-' + sub_layer)
        return chip_layers

    def get_last_out_layer(self):
        last_layer = self.caffe_net_txt.layer[0]
        for idx, layer in enumerate(self.caffe_net_txt.layer):
            if idx < len(self.caffe_net_txt.layer) - 2:
                if layer.type == "QuantConvolution":
                    last_layer = self.caffe_net_txt.layer[idx].name
                    if self.caffe_net_txt.layer[idx+1].type == "Eltwise" or self.caffe_net_txt.layer[idx+1].type == "Pooling":
                        last_layer = self.caffe_net_txt.layer[idx+1].name
                    elif self.caffe_net_txt.layer[idx+2].type == "Pooling" and self.caffe_net_txt.layer[idx+2].pooling_param.pool == 0:
                        last_layer = self.caffe_net_txt.layer[idx+2].name
        return last_layer

    def is_fc_mode(self):
        return self.net_config['layer'][-1]['image_size'] == 14 and self.net_config['layer'][-1]['pooling']

    def forward_caffe(self, image_path, endlayer):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.GTI_IMAGE_WIDTH, self.GTI_IMAGE_HEIGHT))
        b,g,r = cv2.split(img)
        b2 = np.concatenate((b, g, r))
        d2_in = np.reshape(b2, (3, self.GTI_IMAGE_WIDTH, self.GTI_IMAGE_WIDTH)).astype(np.uint8)
        d2_in_clip = np.clip(np.right_shift((np.right_shift(d2_in, 2) + 1),1), 0, 31)
        self.caffe_net.blobs['data'].data[...] = d2_in_clip
        self.caffe_net.forward(end=endlayer)
        caffe_out = np.where(self.caffe_net.blobs[endlayer].data>=0, np.clip(np.floor(self.caffe_net.blobs[endlayer].data+0.5),0, 31), 0)
        return caffe_out

    def forward_chip(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.GTI_IMAGE_WIDTH, self.GTI_IMAGE_HEIGHT))
        b,g,r = cv2.split(img)
        b2 = np.concatenate((b, g, r))
        img_ary= np.asarray(b2).reshape(-1, self.GTI_IMAGE_HEIGHT)

        output_size = self.OUTPUT_CHANNELS * self.OUTPUT_IMAGE_SIZE * self.OUTPUT_IMAGE_SIZE
        chip_res = self.GTIMODEL.GtiEvaluate(img_ary, self.GTI_IMAGE_WIDTH,self.GTI_IMAGE_HEIGHT,self.INPUT_CHANNELS)
        chip_res = chip_res[:output_size]
        chip_out = np.reshape(chip_res, (self.OUTPUT_CHANNELS, self.OUTPUT_IMAGE_SIZE, self.OUTPUT_IMAGE_SIZE))
        return chip_out

    def dump_caffe_layers(self, image_path):
        self.caffe_layers = self.get_caffe_layers()
        self.forward_caffe(image_path, self.caffe_layers[-1])
        for layer in self.caffe_layers:
            layer_feature = np.where(self.caffe_net.blobs[layer].data>=0, np.clip(np.floor(self.caffe_net.blobs[layer].data+0.5),0, 31), 0)
            if self.is_fc_mode(): layer_feature *= 8
            layer_feature_flatten = layer_feature.flatten().astype(np.uint8)
            bin_file = os.path.join(self.caffe_output_dir, layer) + ".bin"
            layer_feature_flatten.tofile(bin_file)

    def dump_chip_layers(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.GTI_IMAGE_WIDTH, self.GTI_IMAGE_HEIGHT))
        b,g,r = cv2.split(img)
        b2 = np.concatenate((b, g, r))
        d2_in = np.reshape(b2, (3, self.GTI_IMAGE_WIDTH, self.GTI_IMAGE_WIDTH)).astype(np.uint8)
        d2_in.tofile(self.img_bin)
        self.chip_layers = self.get_chip_layers()
        call = make_call_with(self.log_fname)
        call("""
        cd {} && \
        GTI_LOG_LEVEL=9 \
        {} \
        {} \
        {} \
        >> {log} 2>>{log}
        """.format(
            self.chip_output_dir,
            os.path.join(os.path.dirname(os.path.realpath(os.path.abspath(__file__))), 'liteDemo'),
            self.gti_model,
            self.img_bin,
            log=self.log_fname)
        )

    def match_image(self):
        #check net_config for the learning modes
        for layer in self.net_config['layer']:
            if 'learning' not in layer or not layer['learning']:
                sys.exit("Please use the model with 'learning=true' enabled for all the layers in order to match layer by layer!")
        self.dump_caffe_layers(self.evaluate_path)
        self.dump_chip_layers(self.evaluate_path)
        if len(self.caffe_layers) != len(self.chip_layers):
            sys.exit("caffe layers and chip layers not match! Please convert the model with 'learning=true' for all the layers.")
        all_match = True
        for i in range(len(self.caffe_layers)):
            caffefile = os.path.join(self.caffe_output_dir, self.caffe_layers[i] + ".bin")
            chipfile = os.path.join(self.chip_output_dir, self.chip_layers[i] + ".bin")
            if not filecmp.cmp(caffefile, chipfile):
                all_match = False
                print(self.caffe_layers[i] + " does not match chip output!") 
        if all_match:
            print("caffe output matches chip output for all the layers!")  

    def match_images(self):
        #check net_config for the learning modes
        for layer in self.net_config['layer']:
            if 'learning' in layer and layer['learning']:
                sys.exit("Please use the model with 'learning=false' for all the layers in order to run batch testing!")
        self.GTIMODEL = gtilib.GtiModel(self.gti_model)
        endlayer = self.get_last_out_layer()
        match_count = 0
        image_count = 0
        for image_name in os.listdir(self.evaluate_path):
            image_path = os.path.join(self.evaluate_path, image_name)
            image_count += 1
            caffe_out = self.forward_caffe(image_path, endlayer)
            chip_out = self.forward_chip(image_path)
            bit_diff = (caffe_out == chip_out)
            if bit_diff.all():
                match_count += 1
                print("Comparing image %s: %r"%(image_path, bit_diff.all()))
            else:
                output_size = self.OUTPUT_CHANNELS * self.OUTPUT_IMAGE_SIZE * self.OUTPUT_IMAGE_SIZE
                bit_match_ratio = np.sum(bit_diff)/float(output_size)    
                print("Comparing image %s: %r(%3f match)"%(image_path, bit_diff.all(), bit_match_ratio))
        print("Total Images: {:5d}, Match Count: {:.3f}, Match Ratio: {:.3f}".format(image_count, match_count, float(match_count)/image_count))

    def convert_bin_txt(self, bin_file, dim):
        TILE_SIZE = 14
        x = np.fromfile(bin_file, dtype=np.uint8) 
        channels, rows, cols = dim                                                                    
        x = x.reshape(channels, rows, cols)
        txt_dir = bin_file.split(".")[0]                                                                  
        if os.path.exists(txt_dir):
            shutil.rmtree(txt_dir)
        os.mkdir(txt_dir)
        for channel in range(channels):
            with open(os.path.join(txt_dir, "c_{}.out".format(channel+1)), "w") as f:
                row_idx = 0
                for row in range(0, rows, TILE_SIZE):
                    col_idx = 0
                    for col in range(0, cols, TILE_SIZE):
                        f.write("blk_i = {:2d} blk_j = {:2d}\n".format(row_idx, col_idx)) 
                        np.savetxt(f, x[channel, row:row+TILE_SIZE, col:col+TILE_SIZE], fmt="%2d")
                        col_idx += 1
                    row_idx += 1

def main(args):
    bitMatch = BitMatch(args)
    if os.path.isdir(args.evaluate_path):
        bitMatch.match_images()
    elif os.path.isfile(args.evaluate_path):
        bitMatch.match_image()
    else:
        sys.exit("evaluate path error!")

def gen_txts_parse_args(argv):
    parser = ArgumentParser(description="pass arguments for bit matching")
    parser.add_argument('caffe_prototxt', help='the net definition prototxt')
    parser.add_argument('caffe_model', help='the weights caffemodel')
    parser.add_argument('net_json', help='net.json')
    parser.add_argument('gti_model', help='the conversion out.model for chip use')
    parser.add_argument('evaluate_path', help='image or image dir path')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('log_fname', help='output text log location, optional', nargs='?', default=None)
    args = parser.parse_args(['_' if x == '-o' else x for x in argv[1:]])
    return args

if __name__ == '__main__':
    main(gen_txts_parse_args(sys.argv))