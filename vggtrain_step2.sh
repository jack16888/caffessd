#!/bin/sh
build/tools/caffe train --solver=step2/solver_face.prototxt --weights=step2/pretrain_iter_160000.caffemodel --gpu 0 2>&1 | tee step2/log.log
#build/tools/caffe train --solver=step2/solver_face.prototxt --weights=step2/pretrain_iter_200000.caffemodel --gpu 0 2>&1 | tee step2/log.log
