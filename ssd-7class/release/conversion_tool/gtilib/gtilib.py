"""
Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
See LICENSE file in the project root for full license information.
"""
from ctypes import *
import numpy as np
import os, sys, platform
import struct

from numpy.ctypeslib import ndpointer

if(platform.system() == "Linux"):
    dllFilename=os.path.join(os.path.dirname(__file__), "libftd3xx.so.0.5.21")
    FTDILib = CDLL(dllFilename)
    dllFilename=os.path.join(os.path.dirname(__file__), "libGTILibrary.so")
    gtiSdkLib = CDLL(dllFilename)
else:
    dllFilename=os.path.join(os.path.dirname(__file__), "FTD3XX.dll")
    FTDILib = CDLL(dllFilename)
    dllFilename=os.path.join(os.path.dirname(__file__), "GTILibrary.dll")
    gtiSdkLib = CDLL(dllFilename)


def GtiComposeModelFile(jsonFilename, modelFilename):
    gtiSdkLib.GtiComposeModelFile.argtypes=(c_char_p, c_char_p)
    return gtiSdkLib.GtiComposeModelFile(jsonFilename.encode('ascii'), modelFilename.encode('ascii'))


class GtiModel(object):
    def __init__(self, modelName):
        gtiSdkLib.GtiCreateModel.argtypes=(c_char_p,)
        gtiSdkLib.GtiCreateModel.restype = c_ulonglong
        self.obj = gtiSdkLib.GtiCreateModel(modelName.encode('ascii'))
    	
    def GtiImageEvaluate(self, image, w, h, d):
        gtiSdkLib.GtiImageEvaluate.argtypes=(c_ulonglong, c_char_p, c_int, c_int, c_int)
        gtiSdkLib.GtiImageEvaluate.restype = c_char_p      
	return gtiSdkLib.GtiImageEvaluate(self.obj, image, w, h, d)

    def GtiImageEvaluate_oneFrame(self, image, w, h, d,outLen):
        gtiSdkLib.GtiImageEvaluate.argtypes=(c_ulonglong, c_char_p, c_int, c_int, c_int)
        gtiSdkLib.GtiImageEvaluate.restype = POINTER(c_float*outLen)
        out = gtiSdkLib.GtiImageEvaluate(self.obj, image, w, h, d)    
	return [int(i) for i in out.contents] 
    def GtiEvaluate(self, image,w,h,d):
        # -----------input----------
	#      image:     numpary array, numpy.int32
        #          w:     int, weight of image
        #          h:     int, height of image
        #          d:     int, depth of image
        # ---------- -ouput----------
	# buffer_out:     pointer of output
        #   size_out:     int, size of buffer
        #      w_out:     int, weight of image
        #      h_out:     int, height of image
        #      d_out:     int, depth of image        	     
	input_buffer = bytes()
        input_buffer = struct.pack('iiii',w,h,d,0)
        img_buf = image.ctypes.data
        input_buffer +=struct.pack('q',img_buf)

        gtiSdkLib.GtiEvaluate.argtypes=(c_ulonglong, c_char_p)
        out_ref = gtiSdkLib.GtiEvaluate(self.obj, input_buffer)
        
	w_out = cast(out_ref, POINTER(c_int)).contents.value
	h_out = cast(out_ref + 4, POINTER(c_int)).contents.value
        d_out = cast(out_ref + 8, POINTER(c_int)).contents.value
        s_out = cast(out_ref + 12, POINTER(c_int)).contents.value
        buffer_out = cast(out_ref + 16, POINTER(c_int)).contents.value
        size_out = cast(out_ref + 24, POINTER(c_int)).contents.value

        INTP = POINTER(c_float)
        res = cast(buffer_out, INTP)        
        #print(w_out,h_out,d_out,s_out,size_out,res[0:10])
	#return buffer_out,size_out,w_out,h_out,d_out
	return res   

    def GtiDestroyModel(self):
        gtiSdkLib.GtiDestroyModel.argtypes=(c_ulonglong,)
        gtiSdkLib.GtiDestroyModel.restype = c_int
        return gtiSdkLib.GtiDestroyModel(self.obj)
    '''
    def GtiEvaluate(self, input):
        #Input and output of this function is class GtiTensor
        gtiSdkLib.GtiEvaluate.argtypes = (c_ulonglong, c_char_p)
        return gtiSdkLib.GtiEvaluate(self.obj, input)
    '''
    


class GtiDevice(object):
    def __init__(self, DeviceType, FilterFileName, ConfigFileName):
        gtiSdkLib.GtiDeviceCreate.restype = c_ulonglong
        self.obj = gtiSdkLib.GtiDeviceCreate(DeviceType, FilterFileName.encode('ascii'), ConfigFileName.encode('ascii'))

    def OpenDevice(self, deviceName):
        gtiSdkLib.GtiOpenDevice.argtypes = (c_ulonglong, c_char_p)
        return gtiSdkLib.GtiOpenDevice(self.obj, deviceName.encode('ascii'))

    def CloseDevice(self):
        gtiSdkLib.GtiCloseDevice.argtypes = (c_ulonglong,)
        gtiSdkLib.GtiCloseDevice(self.obj)

    def GtiDeviceRelease(self):
        gtiSdkLib.GtiDeviceRelease.argtypes = (c_ulonglong,)
        gtiSdkLib.GtiDeviceRelease(self.obj)

    def Initialization(self):
        gtiSdkLib.GtiInitialization.argtypes = (c_ulonglong,)
        return gtiSdkLib.GtiInitialization(self.obj)

    def SelectNetwork(self, networkId):
        gtiSdkLib.GtiSelectNetwork.argtypes = (c_ulonglong, c_int)
        return gtiSdkLib.GtiSelectNetwork(self.obj, networkId)

    def SendImage(self, inBuffer, inLength):
        gtiSdkLib.GtiSendImage.argtypes = (c_ulonglong, c_char_p, c_int)
        return gtiSdkLib.GtiSendImage(self.obj, inBuffer, inLength)

    def SendImageFloat(self, inBuffer, inLength):
        gtiSdkLib.GtiSendImageFloat.argtypes = (c_ulonglong, c_char_p, c_int)
        return gtiSdkLib.GtiSendImageFloat(self.obj, inBuffer, inLength)

    def GetOutputData(self, outBuffer, outLength):
        gtiSdkLib.GtiGetOutputData.argtypes = (c_ulonglong, c_char_p, c_int)
        return gtiSdkLib.GtiGetOutputData(self.obj, outBuffer, outLength)

    def GetOutputDataFloat(self, outBuffer, outLength):
        gtiSdkLib.GtiGetOutputDataFloat.argtypes = (c_ulonglong, c_char_p, c_int)
        return gtiSdkLib.GtiGetOutputDataFloat(self.obj, outBuffer, outLength)

    def HandleOneFrame(self, inBuff, inLen, outBuff, outLen):
        gtiSdkLib.GtiHandleOneFrame.argtypes = (c_ulonglong, c_char_p, c_int, c_char_p, c_int)
        return gtiSdkLib.GtiHandleOneFrame(self.obj, inBuff, inLen, outBuff, outLen)

    def HandleOneFrameFloat(self, inBuff, inLen, outBuff, outLen):
        gtiSdkLib.GtiHandleOneFrameFloat.argtypes = (c_ulonglong, c_char_p, c_int, c_char_p, c_int)
        return gtiSdkLib.GtiHandleOneFrameFloat(self.obj, inBuff, inLen, outBuff, outLen)

    def HandleOneFrameLt(self, inBuff, inLen, outBuff, outLen):
        gtiSdkLib.GtiHandleOneFrameFloat.argtypes = (c_ulonglong, c_char_p, c_int, c_char_p, c_int)
        ret = gtiSdkLib.GtiHandleOneFrameFloat(self.obj, inBuff, inLen, outBuff, outLen)
        outList = np.frombuffer(outBuff, dtype='float32', count=outLen, offset=0)
        return ret, outList

    def GetOutputLength(self):
        gtiSdkLib.GtiGetOutputLength.argtypes = (c_ulonglong,)
        return gtiSdkLib.GtiGetOutputLength(self.obj)


    def OpenDeviceAndInit(self, deviceName):
        # Open device.
        gtiSdkLib.GtiOpenDevice.argtypes = (c_ulonglong, c_char_p)
        gtiSdkLib.GtiOpenDevice(self.obj, deviceName.encode('ascii'))
        # Initialization.
        gtiSdkLib.GtiInitialization.argtypes = (c_ulonglong,)
        ret = gtiSdkLib.GtiInitialization(self.obj)
        return ret

    def ReopenDevice(self, deviceName):
        gtiSdkLib.GtiCloseDevice.argtypes = (c_ulonglong,)
        gtiSdkLib.GtiCloseDevice(self.obj)
        gtiSdkLib.GtiOpenDevice.argtypes = (c_ulonglong, c_char_p)
        return gtiSdkLib.GtiOpenDevice(self.obj, deviceName.encode('ascii'))
