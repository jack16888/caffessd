#!/bin/sh
#build/tools/caffe train --solver=step3/solver_face.prototxt --weights=step3/pretrain_iter_170000_QuantReLU.caffemodel --gpu 0 2>&1 | tee step3/log.log
build/tools/caffe train --solver=step3/solver_face.prototxt --weights=step3/pretrain_iter_195000_QuantReLU.caffemodel --gpu 0 2>&1 | tee step3/log.log
