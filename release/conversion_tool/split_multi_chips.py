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
This script contains functions to split a single filter.txt/bias.txt/net.json into multiple filter/bias/json groups for multi-chip solution 

  --Usage: After filter.txt/bias.txt/net.json files have been dumped from a big model. 
           Speficially used for the mult-chip solution to generate multiple filter/bias/json groups

  --Details:  
        # Input:  a single filter.txt/bias.txt/net.json directly dumped from a trained checkpoint of a big model (e.g., ResNet-50, ResNet-101)
        # Output: Multiple filter.txt/bias.txt/net.json groups, each of which will be used for generating one .dat chip file
                  Each .dat file will be loaded onto on chip nested in a multi-chip board. 
  --Functions: 
    (1) SplitJson 
        Split a single .json file into multiple json file 
    (2) SplitFltTxt 
        Split a single filter.txt file into multiple filter.txt file 
    (3) SplitBiasTxt 
        Split a single bias.txt file into multiple bias.txt file 

  --Notes: This is an example script for GTI-ResNet. Users need to modify the code for other chip-splitting strategies   
 
"""

import sys
import json 
import numpy as np
from collections import namedtuple

MAX_LAYER_NUM = 6
kernel_size = 9
Layer = namedtuple('Layer', 'major_layer input_channels \
                             output_channels sublayer_number \
                             compression_ratio')

resnet50_chip = [2, 3, 4]
resnet50_split_chip = {
   "2": {'split_point': [7], 'pooling_method': [0, 1]},
   "3": {'split_point': [5, 7], 'pooling_method': [0, 0, 1]},
   "4": {'split_point': [5, 6, 7], 'pooling_method': [0, 0, 0, 1]}
}

resnet110_chip = [4, 5, 6, 7, 8, 9 ,10]
resnet110_split_chip = {
   "4": {'split_point': [7, 9, 13], \
         'pooling_method': [0, 0, 0, 1]},
   "5": {'split_point': [7, 9, 11, 13], \
         'pooling_method': [0, 0, 0, 0, 1]},
   "6": {'split_point': [5, 7, 9, 11, 13], \
         'pooling_method': [0, 0, 0, 0, 0, 1]},
   "7": {'split_point': [5, 6, 7, 9, 11, 13], \
         'pooling_method': [0, 0, 0, 0, 0, 0, 1]},
   "8": {'split_point': [5, 6, 7, 8, 9, 11, 13], \
         'pooling_method': [0, 0, 0, 0, 0, 0, 0, 1]},
   "9": {'split_point': [5, 6, 7, 8, 9, 10, 11, 13], \
         'pooling_method': [0, 0, 0, 0, 0, 0, 0, 0, 1]},
   "10:": {'split_point': [5, 6, 7, 8, 9, 10, 11, 12, 13], \
           'pooling_method': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
}

def _LayerParser(json_file):
   
    with open(json_file, 'r') as f:
       net_arch = json.load(f)
       m_layer_list = []     
       for m_layer in net_arch['layer']:
          major_layer = m_layer['major_layer'] 
          input_channels = m_layer['input_channels'] 
          output_channels = m_layer['output_channels'] 
          sublayer_number = m_layer['sublayer_number']

          if m_layer['coef_bits']==1 or m_layer['coef_bits']==2:
             compression_ratio = 4 
          elif m_layer['coef_bits']==3 or m_layer['coef_bits']==5: 
             compression_ratio = 2 
          else:
             compression_ratio = 1 
              
          m_layer_list.append(Layer(major_layer, input_channels, \
                              output_channels, sublayer_number, \
                              compression_ratio)) 

    return m_layer_list

def ChipParser(model, json_file, chip_num):

    m_layer_list = _LayerParser(json_file)
    layer_range = range(0, len(m_layer_list))  

    # TO DO: Hard coded at this moment
    if ('resnet50' in model and chip_num in resnet50_chip) \
       or ('resnet110' in model and chip_num in resnet110_chip):
       
       if 'resnet50' in model: 
          split_point = resnet50_split_chip[str(chip_num)]['split_point']
       elif 'resnet110' in model:
          split_point = resnet110_split_chip[str(chip_num)]['split_point']
       else:
          raise NotImplementedError("Model %s can not support chip number of %d"%(model, chip_num)) 
   
       layer_split = np.split(layer_range, split_point)

       layer2chip = {}
       
       flt_split_point = [] 
       bias_split_point = []
       flt_line_count = 0 
       bias_line_count = 0 
       for chip_idx in range(chip_num):               
           flt_split_point.append(flt_line_count)
           bias_split_point.append(bias_line_count)

           chip_name = "chip%d"%chip_idx
           layer2chip[chip_name] = [m_layer_list[i] for i in layer_split[chip_idx]]

           for layer_idx, m_layer in enumerate(layer2chip[chip_name]):  
              major_layer_id = m_layer.major_layer
              in_ch = m_layer.input_channels
              out_ch = m_layer.output_channels
              sublayer_num = m_layer.sublayer_number

              for sublayer_idx in range(sublayer_num):
                  if sublayer_idx == 0:
                     line_sublayer = in_ch*out_ch*kernel_size 
                  else:
                     line_sublayer = out_ch*out_ch*kernel_size 
                 
                  flt_line_count += line_sublayer
                  bias_line_count += out_ch

       return layer2chip, flt_split_point, bias_split_point 
   
    raise NotImplementedError("Model %s can not support chip number of %d"%(model, chip_num)) 


def SplitJson(json_file, layer2chip):

    chip_num = len(layer2chip) 
    layer_range = range(0, sum(len(v) for v in layer2chip.itervalues()))
    layer_split = np.split(layer_range, resnet50_split_chip[str(chip_num)]['split_point'])
    pooling_method = resnet50_split_chip[str(chip_num)]['pooling_method']

    for chip_idx in range(chip_num):

       net_arch_chip = dict()
       with open(json_file, 'r') as f:
          net_arch = json.load(f)

       chip_name = "chip%d"%chip_idx
       json_path = json_file[:-5] + "_chip%d.json"%chip_idx

       m_layer_number = len(layer2chip[chip_name])
       net_arch_chip['model'] = net_arch['model']
       net_arch_chip['model'][0]['MajorLayerNumber'] = np.min([m_layer_number, MAX_LAYER_NUM])
       net_arch_chip['model'][0]['ChipNumber'] = 1
       
       if pooling_method[chip_idx] == 1:
          net_arch_chip['model'][0]['SamplingMethod'] = 1

       m_layer_id = [m_layer.major_layer for m_layer in layer2chip[chip_name]]
       if len(m_layer_id)==MAX_LAYER_NUM+1:
          layer7_num = int(net_arch['layer'][MAX_LAYER_NUM-1]['sublayer_number'])
          layer8_num = int(net_arch['layer'][MAX_LAYER_NUM]['sublayer_number'])
          sublayer_number_sum = layer7_num + layer8_num
          m_layer_id = m_layer_id[:-1]

          if sublayer_number_sum < 13:
             #print('Combining two repeating modules into one major layer')
             net_arch['layer'][MAX_LAYER_NUM-1]['sublayer_number'] = sublayer_number_sum
             net_arch['layer'][MAX_LAYER_NUM-1]['scaling'] += net_arch['layer'][MAX_LAYER_NUM]['scaling']
             resnet_shortcut_start_layers_7 = net_arch['layer'][MAX_LAYER_NUM-1]['resnet_shortcut_start_layers']
             resnet_shortcut_start_layers_8 = net_arch['layer'][MAX_LAYER_NUM]['resnet_shortcut_start_layers']

             if resnet_shortcut_start_layers_7 is not None:
                 if resnet_shortcut_start_layers_8 is None:
                     net_arch['layer'][MAX_LAYER_NUM - 1]['resnet_shortcut_start_layers'] = resnet_shortcut_start_layers_7
                 else:
                     resnet_shortcut_start_layers_8 = map(lambda x : x + layer7_num, resnet_shortcut_start_layers_8)
                     net_arch['layer'][MAX_LAYER_NUM - 1]['resnet_shortcut_start_layers'] = resnet_shortcut_start_layers_7 + resnet_shortcut_start_layers_8
             else:
                 if resnet_shortcut_start_layers_8 is not None:
                     resnet_shortcut_start_layers_8 = map(lambda x : x + layer7_num, resnet_shortcut_start_layers_8)
                     net_arch['layer'][MAX_LAYER_NUM - 1]['resnet_shortcut_start_layers'] = resnet_shortcut_start_layers_8
             
          else:
            raise ValueError('Layer definition wrong: major layer number exceeds the limit')
       elif len(m_layer_id) > MAX_LAYER_NUM+1:
          raise ValueError('Layer definition wrong: major layer number exceeds the limit')
          
       net_arch_chip['layer'] = [net_arch['layer'][layer_id] \
                                 for layer_id in layer_range \
                                    if (layer_id + 1) in m_layer_id]
       
       for idx in range(len(m_layer_id)):
          if idx < MAX_LAYER_NUM:
             net_arch_chip['layer'][idx]['major_layer'] = idx + 1 

       with open(json_path, 'w') as f:
          json.dump(net_arch_chip, f, indent=4, separators=(',', ': '), sort_keys=True)


def SplitFltTxt(flt_file, flt_split_point):

    flt = np.loadtxt(flt_file, dtype=float)
    chip_num = len(flt_split_point) 
    for chip_idx in range(chip_num): 
       flt_path = flt_file[:-4] + "_chip%d.txt"%chip_idx
       flt_split = np.split(flt, flt_split_point[1:])
       flt_for_chip = flt_split[chip_idx]
       np.savetxt(flt_path, flt_for_chip, fmt= '%.16e', delimiter='\n')


def SplitBiasTxt(bias_file, bias_split_point):

    bias = np.loadtxt(bias_file, dtype=float)
    chip_num = len(bias_split_point) 
    for chip_idx in range(chip_num): 
       bias_path = bias_file[:-4] + "_chip%d.txt"%chip_idx
       bias_split = np.split(bias, bias_split_point[1:])
       bias_for_chip = bias_split[chip_idx]
       np.savetxt(bias_path, bias_for_chip, fmt= '%.16e', delimiter='\n')


def SplitMultiChips(flt_file, bias_file, json_file, chip_num, model):

    layer2chip, flt_split_point, bias_split_point = ChipParser(model, json_file, chip_num)
    SplitJson(json_file, layer2chip)
    SplitFltTxt(flt_file, flt_split_point)
    SplitBiasTxt(bias_file, bias_split_point)
