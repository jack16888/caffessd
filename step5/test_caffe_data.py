import caffe
import cv2
import numpy as np


deploy = "./net_refined6_deploy.prototxt"
caffe_model = "./step4_iter_10000.caffemodel"
net = caffe.Net(deploy,caffe_model,caffe.TEST)
caffe.set_mode_gpu()

img = cv2.imread("car.jpg")

img = cv2.resize(img, (224, 224))    
b,g,r = cv2.split(img)
b2 = np.concatenate((b, g, r))
b2 = b2.reshape(3,224,224)

b2 = np.clip(np.right_shift((np.right_shift(b2,2) + 1),1), 0, 31)
net.blobs['data'].data[...] = b2
out = net.forward()

pool5 = net.blobs['pool5'].data[...][0]
print(pool5)
print pool5.shape
with open('pool5.txt', 'w') as f:
    for item in pool5.flatten():
        f.write(str(int(item))+'\n')




