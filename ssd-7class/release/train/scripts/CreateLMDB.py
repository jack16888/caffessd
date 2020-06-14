import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb
import sys
import json

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Image Resizing
    img = cv2.resize(img, (img_width, img_height))
    return img


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring()
    )

def CreateLMDB(data_path):
    train_lmdb = os.path.join(data_path, 'train_lmdb')
    val_lmdb = os.path.join(data_path, 'val_lmdb')
    os.system('rm -rf ' + train_lmdb)
    os.system('rm -rf ' + val_lmdb)

    train_data = []
    val_data = []
    for path in glob.glob(data_path + "/train/*"):
        for img in glob.glob(path + '/*'):
            train_data.append(img)

    for path in glob.glob(data_path + "/val/*"):
        for img in glob.glob(path + '/*'):
            val_data.append(img)

    random.shuffle(train_data)

    print "Creating train_lmdb"
    label_dict = {}
    label_count = 0
    in_db = lmdb.open(train_lmdb, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(train_data):
            label_name = img_path.split('/')[-2]
            if label_name not in label_dict:
                label_dict[label_name] = label_count
                label_count += 1
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            label = label_dict[label_name]
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

    print "\nCreating val_lmdb"
    in_db = lmdb.open(val_lmdb, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(val_data):
            label_name = img_path.split('/')[-2]
            if label_name not in label_dict:
                raise ValueError("label not exist!")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            label = label_dict[label_name]
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

    sorted_by_value = sorted(label_dict.items(), key=lambda kv: kv[1])
    return sorted_by_value


def main(argv):
    label_dict = CreateLMDB(argv[1])
    label_file = os.path.join(argv[1], 'labels.txt')
    with open(label_file, 'w') as f:
        for key, value in label_dict:
            f.write(str(value) + ' ' + str(key) + '\n')
    print "\nFinished!"


if __name__ == '__main__':
    main(sys.argv)
