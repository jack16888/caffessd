#!/usr/bin/env python
"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
import os
import sys
import datetime
import shutil
import json
import subprocess
from utils import extract_int_by_string, make_call_with, mktemp
from split_multi_chips import SplitMultiChips

USAGE_STR = "usage:\n\tpython internal_convert.py fitler.txt bias.txt net.json" \
            "{-o} " \
            "[--output_dir=]{path} " \
            "[--debug=]{true|false} " + \
            "\nNote that arguments must be in the correct order except the optional arguments."

# New implmenetation using a single .so wrapped by python subprocess
def internal_convert_once(chip_type, filter_path, bias_path, net_json_path, out_dat_path, out_tb_path, log_fname, temp_dir=None,
                     debug=False):
    OLD_PWD = os.getcwd()
    filter_path = os.path.abspath(filter_path)
    bias_path = os.path.abspath(bias_path)
    net_json_path = os.path.abspath(net_json_path)
    out_dat_path = os.path.abspath(out_dat_path)
    out_tb_path = os.path.abspath(out_tb_path)
    log_fname = os.path.abspath(log_fname)
    temp_dir = os.path.abspath(temp_dir)
    try:
        # cd into my current directory
        os.chdir(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
        call = make_call_with(log_fname)
        call("""
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./ python libgticonfig.py \
    {} \
    {} \
    {} \
    {} \
 -o {} {} {} {}\
    >> {log} 2>> {log}
    """.format(
            filter_path,
            bias_path,
            net_json_path,
            'GTI' + str(chip_type),
            out_dat_path,
            out_tb_path,
            temp_dir,
            '--debug=true' if debug else '',
            log=log_fname
        ))
    finally:
        os.chdir(OLD_PWD)


def internal_convert(filter_path, bias_path, net_json_path, log_fname, output_dir=None, temp_dir=None, debug=False):
    if output_dir == None:
        output_dir = os.path.abspath('output')
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
            pass
    if temp_dir == None:
        temp_dir = os.path.join(output_dir, "debug")
        os.makedirs(temp_dir)

    out_dat_path = os.path.join(output_dir, 'coef.dat')
    out_tb_path = os.path.join(temp_dir, 'coef.tb')

    with open(net_json_path) as jf:
        jj = json.load(jf)
        chip_nums = jj['model'][0]['ChipNumber'] if 'ChipNumber' in jj['model'][0] else 1
        chip_type = jj['model'][0]['ChipType'] if 'ChipType' in jj['model'][0] else 0
        model_type = jj['model'][0]['ModelType'] if "ModelType" in jj['model'][0] else ""

    if chip_nums == 1:
        internal_convert_once(chip_type, filter_path, bias_path, net_json_path, out_dat_path, out_tb_path, log_fname,
                              temp_dir=temp_dir, debug=debug)

    elif chip_nums > 1:
        #Split into multiple chips
        SplitMultiChips(filter_path, bias_path, net_json_path, chip_nums, model_type)
        #Call internal_convert_once multiple times
        for i in range(chip_nums):
            cur_chip_name = "_chip" + str(i)
            cur_filter_path = filter_path[:-4] + cur_chip_name + ".txt"
            cur_bias_path = bias_path[:-4] + cur_chip_name + ".txt"
            cur_net_json_path = net_json_path[:-5] + cur_chip_name + ".json"
            cur_out_dat_path = out_dat_path[:-4] + cur_chip_name + ".dat"
            cur_temp_dir = os.path.join(temp_dir, cur_chip_name)
            try:
                os.makedirs(cur_temp_dir)
            except OSError as e:
                pass
            cur_out_tb_path = os.path.join(cur_temp_dir, "coef"+cur_chip_name+".tb")
            internal_convert_once(chip_type, cur_filter_path, cur_bias_path, cur_net_json_path, cur_out_dat_path,
                                  cur_out_tb_path, log_fname, temp_dir=cur_temp_dir, debug=debug)
    else:
        print("Chip Numbers Error!")
        exit(1)

def main(argv):
    log_fname = os.path.abspath('internal_log.txt')
    if os.path.exists(log_fname):
        os.remove(log_fname)
    with open(log_fname, 'a') as f:
        f.write('\n\nstarting internal processing of {} on {}\n\n'.format(repr(argv), datetime.datetime.utcnow()))

    if len(argv) < 4:
        print(USAGE_STR)
        exit(1)

    OUTPUT_DIR = os.path.abspath('internal_output')
    DEBUG = False
    filter_path = os.path.abspath(argv[1])
    bias_path = os.path.abspath(argv[2])
    net_json_path = os.path.abspath(argv[3])

    if '-o' in argv:
        if len(argv) <= 5 or argv[4] != '-o':
            print(USAGE_STR)
            exit(1)
        opt_argvs = argv[5:]
        for i, opt_argv in enumerate(opt_argvs):
            if '--output_dir' in opt_argv:
                OUTPUT_DIR = os.path.abspath(opt_argvs[i].split('=')[-1])
            if '--debug' in opt_argv:
                if opt_argvs[i].split('=')[-1] == 'True' or opt_argvs[i].split('=')[-1] == 'true':
                    DEBUG = True

    TEMP_DIR = os.path.join(OUTPUT_DIR, 'debug')

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    try:
        os.makedirs(OUTPUT_DIR)
        os.makedirs(TEMP_DIR)
    except OSError as e:
        pass

    internal_convert(filter_path, bias_path, net_json_path, log_fname, output_dir = OUTPUT_DIR, temp_dir = TEMP_DIR, debug = DEBUG)

    if not DEBUG:
        shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    main(sys.argv)
