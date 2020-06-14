"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
from ctypes import *
import os, sys, platform

if (platform.system() == "Linux"):
    dllFilename = os.path.join(os.path.dirname(__file__), "libgticonfig2801.so")
    gtiConfigLib2801 = CDLL(dllFilename)
    dllFilename = os.path.join(os.path.dirname(__file__), "libgticonfig2803.so")
    gtiConfigLib2803 = CDLL(dllFilename)
    dllFilename = os.path.join(os.path.dirname(__file__), "libgticonfig5801.so")
    gtiConfigLib5801 = CDLL(dllFilename)
else:
    sys.exit(1)


def convert(jsonf, filterf, biasf, chip_type, datf, tbf, tmpdir, debug=False):
    jsonf = jsonf.encode('utf-8')
    filterf = filterf.encode('utf-8')
    biasf = biasf.encode('utf-8')
    datf = datf.encode('utf-8')
    tbf = tbf.encode('utf-8')
    tmpdir = tmpdir.encode('utf-8')
    if chip_type == 'GTI2801':
        gtiConfigLib2801.GtiConvertInternalToSDK.argtypes = (
        c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_bool)
        gtiConfigLib2801.GtiConvertInternalToSDK(jsonf, filterf, biasf, chip_type.encode('utf-8'), datf, tbf, tmpdir, debug)
    elif chip_type == 'GTI2803':
        gtiConfigLib2803.GtiConvertInternalToSDK.argtypes = (
        c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_bool)
        gtiConfigLib2803.GtiConvertInternalToSDK(jsonf, filterf, biasf, chip_type.encode('utf-8'), datf, tbf, tmpdir, debug)
    elif chip_type == 'GTI5801':
        gtiConfigLib5801.GtiConvertInternalToSDK.argtypes = (
        c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_bool)
        gtiConfigLib5801.GtiConvertInternalToSDK(jsonf, filterf, biasf, chip_type.encode('utf-8'), datf, tbf, tmpdir, debug)
    else:
        print('Invalid chip type ' + chip_type)
        sys.exit(1)


if __name__ == '__main__':
    convert(filterf=sys.argv[1], biasf=sys.argv[2], jsonf=sys.argv[3], chip_type=sys.argv[4], datf=sys.argv[6],
            tbf=sys.argv[7], tmpdir=sys.argv[8], debug=(len(sys.argv) > 9))
