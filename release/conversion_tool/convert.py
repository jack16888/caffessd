#!/usr/bin/env python
"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
import os
import sys
import subprocess
import datetime
import shutil
import json
from utils import make_call_with
from update_fullmodel import update_fullmodel
from internal_convert import internal_convert

USAGE_STR = "usage:\n\tpython convert.py *.prototxt *.caffemodel network*.json fullmodel_def*.json " \
            "{-o} " \
            "[--label=]{path} " \
            "[--shift_max=]{0..15} " \
            "[--input_scale=]{0.1215 (31/255)} " \
            "[--output_dir=]{path} " \
            "[--evaluate=]{path} " \
            "[--debug=]{true|false} [--edit_gain=]{true|false} " + \
            "\nNote that arguments must be in the correct order except the optional arguments."

def new_api(argv):
    if len(argv) < 5:
        print(USAGE_STR)
        exit(1)

    PROTO_INFILE = os.path.abspath(argv[1])
    MODEL_INFILE = os.path.abspath(argv[2])
    NET_INFILE = os.path.abspath(argv[3])
    FULLMODEL_INFILE = os.path.abspath(argv[4])
    LABELS_INFILE = 'labels.txt'
    OUTPUT_DIR = os.path.abspath('output')
    CONVERSION_LOGFILE = None
    INPUT_SCALE = 1.0
    DEBUG = False
    EDIT_GAIN = False
    EVALUATE_PATH = None
    SHIFT_MAX = 0
    if '-o' in argv:
        if len(argv) <= 6 or argv[5] != '-o':
            print(USAGE_STR)
            exit(1)
        opt_argvs = argv[6:]

        for i, opt_argv in enumerate(opt_argvs):
            if '--output_dir' in opt_argv:
                OUTPUT_DIR = os.path.abspath(opt_argvs[i].split('=')[-1])
        for i, opt_argv in enumerate(opt_argvs):
            if '--label' in opt_argv:
                LABELS_INFILE = os.path.abspath(opt_argvs[i].split('=')[-1])
            if '--shift_max' in opt_argv:
                SHIFT_MAX = opt_argvs[i].split('=')[-1]
            if '--input_scale' in opt_argv:
                INPUT_SCALE = opt_argvs[i].split('=')[-1]
            if '--debug' in opt_argv:
                if opt_argvs[i].split('=')[-1] == 'True' or opt_argvs[i].split('=')[-1] == 'true':
                    DEBUG = True
            if '--edit_gain' in opt_argv:
                if opt_argvs[i].split('=')[-1] == 'True' or opt_argvs[i].split('=')[-1] == 'true':
                    EDIT_GAIN = True
                    CONVERSION_LOGFILE = OUTPUT_DIR + "/conversion.log"
            if '--evaluate' in opt_argv:
                EVALUATE_PATH = os.path.abspath(opt_argvs[i].split('=')[-1])

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        pass

    with open(NET_INFILE) as jf:
        net_config = json.load(jf)
        CHIP_ID = net_config['model'][0]['ChipType'] if 'ChipType' in net_config['model'][0] else 0

    if(CHIP_ID == 0):
        print("Chip Type Not Found!")
        exit(1)
    WEIGHT_BITS = 8 if CHIP_ID == 5801 else 12
    if SHIFT_MAX == 0: # use default shift max, if user not provided
        SHIFT_MAX = 15 if CHIP_ID == 5801 else 13

    do_convert(PROTO_INFILE, MODEL_INFILE, NET_INFILE, FULLMODEL_INFILE, LABELS_INFILE, WEIGHT_BITS, SHIFT_MAX,
               INPUT_SCALE, OUTPUT_DIR, EVALUATE_PATH, CONVERSION_LOGFILE, debug=DEBUG, edit_gain=EDIT_GAIN)

def do_convert(PROTO_INFILE, MODEL_INFILE, NET_INFILE, FULLMODEL_INFILE, LABELS_INFILE, WEIGHT_BITS, SHIFT_MAX,
               INPUT_SCALE, OUTPUT_DIR, EVALUATE_PATH=None, CONVERSION_LOGFILE=None, tempdir=None, debug=False,
               edit_gain=False):
    log_fname = os.path.join(os.path.abspath(OUTPUT_DIR), "log.txt")
    with open(log_fname, 'a') as f:
        f.write('\n\nstarting processing of do_convert on {}\n\n'.format(datetime.datetime.utcnow()))

    OLD_PWD = os.getcwd()
    PROTO_INFILE = os.path.abspath(PROTO_INFILE)
    MODEL_INFILE = os.path.abspath(MODEL_INFILE)
    NET_INFILE = os.path.abspath(NET_INFILE)
    FULLMODEL_INFILE = os.path.abspath(FULLMODEL_INFILE)
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    EVALUATE_PATH = os.path.abspath(EVALUATE_PATH) if EVALUATE_PATH else None
    CONVERSION_LOGFILE = os.path.abspath(CONVERSION_LOGFILE) if CONVERSION_LOGFILE else None

    try:
        os.chdir(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
        call = make_call_with(log_fname)
        TEMP_DIR = os.path.join(OUTPUT_DIR, "debug")
        os.makedirs(os.path.join(TEMP_DIR, 'common_format'))
        call("""
python generate_params.py \
    {} \
    {} \
    {} \
    {} \
       \
    {} \
    {} \
    {} \
    {} \
    {} \
    {} \
       \
 -o {} \
    {} \
    {} \
    {} \
    {} \
    {} \
    >> {log} 2>> {log}
    """.format(
            PROTO_INFILE,
            MODEL_INFILE,
            NET_INFILE,
            FULLMODEL_INFILE,

            WEIGHT_BITS,
            SHIFT_MAX,
            INPUT_SCALE,
            edit_gain,
            debug,
            EVALUATE_PATH,

            TEMP_DIR,
            os.path.join(TEMP_DIR, 'common_format', 'filter.txt'),
            os.path.join(TEMP_DIR, 'common_format', 'bias.txt'),
            os.path.join(TEMP_DIR, 'common_format', 'net.json'),
            os.path.join(TEMP_DIR, 'fc.bin'),
            CONVERSION_LOGFILE if CONVERSION_LOGFILE is not None else '',
            log=log_fname
        ))
        internal_convert(
            os.path.join(TEMP_DIR, 'common_format', 'filter.txt'),
            os.path.join(TEMP_DIR, 'common_format', 'bias.txt'),
            os.path.join(TEMP_DIR, 'common_format', 'net.json'),
            log_fname=log_fname,
            output_dir=OUTPUT_DIR,
            temp_dir=TEMP_DIR,
            debug=debug,
        )
        # modify fullmodel.json to support sdk5.0
        update_fullmodel(
            FULLMODEL_INFILE,
            os.path.join(TEMP_DIR, 'common_format'),
            EVALUATE_PATH is not None
        )
        # now run modelTool
        call("""
mv {} {} && \
cp {} {} && \
cd {} && \
{} modelenc \
    {} >> {log} 2>>{log} && \
    mv {} {}
    """.format(
            os.path.join(OUTPUT_DIR, 'coef*.dat'),
            TEMP_DIR,
            LABELS_INFILE,
            os.path.join(TEMP_DIR, 'labels.txt'),
            TEMP_DIR,
            os.path.join(os.path.dirname(os.path.realpath(os.path.abspath(__file__))), 'modelTool'),
            os.path.join(TEMP_DIR, 'common_format', 'fullmodel.json'),
            os.path.join(TEMP_DIR, 'common_format', 'fullmodel.json.gti'),
            os.path.join(OUTPUT_DIR, 'out.model'),
            log=os.path.join(os.path.dirname(os.path.realpath(os.path.abspath(__file__))), log_fname)
        ))
        # do the bit match evaluation for the converted model
        if EVALUATE_PATH is not None:
            GTI_MODEL = os.path.join(OUTPUT_DIR, "out.model")
            NET_FILE = os.path.join(TEMP_DIR, 'common_format', 'net.json')
            do_evaluate(PROTO_INFILE, MODEL_INFILE, NET_FILE, GTI_MODEL, EVALUATE_PATH)
    finally:
        os.chdir(OLD_PWD)
        if not debug:
            shutil.rmtree(TEMP_DIR)

def do_evaluate(PROTO_INFILE, MODEL_INFILE, NET_INFILE, GTI_MODEL, EVALUATE_PATH):
    OUTPUT_DIR = os.path.abspath('evaluate_output')
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        pass

    log_fname = os.path.join(OUTPUT_DIR, "evaluate_log.txt")
    with open(log_fname, 'a') as f:
        f.write('\n\nstarting processing of do_evaluate on {}\n\n'.format(datetime.datetime.utcnow()))
    PROTO_INFILE = os.path.abspath(PROTO_INFILE)
    CAFFE_MODEL_INFILE = os.path.abspath(MODEL_INFILE)
    NET_INFILE = os.path.abspath(NET_INFILE)
    GTI_MODEL_INFILE = os.path.abspath(GTI_MODEL)
    EVALUATE_PATH = os.path.abspath(EVALUATE_PATH)
    call = make_call_with(log_fname)
    call("""
        python bit_match.py \
            {} \
            {} \
            {} \
            {} \
            {} \
            {} \
            {} \
            >> {log} 2>>{log}
        """.format(
        PROTO_INFILE,
        CAFFE_MODEL_INFILE,
        NET_INFILE,
        GTI_MODEL_INFILE,
        EVALUATE_PATH,
        OUTPUT_DIR,
        log_fname,
        log=log_fname))

def main(argv):
    new_api(argv)

if __name__ == '__main__':
    main(sys.argv)
