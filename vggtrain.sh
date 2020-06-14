#!/bin/sh
#build/tools/caffe train --solver=step1/solver_face.prototxt  --gpu 0 2>&1 | tee step1/log.log
build/tools/caffe train --solver=step1/solver_face.prototxt --weights=step1/pretrain_iter_145000.caffemodel --gpu 0 2>&1 | tee step1/log.log
#build/tools/caffe train --solver=step1/solver_face.prototxt  --weights=step1/pretrain_iter_105000.caffemodel  --gpu 0 2>&1 | tee step1/log.log
#build/tools/caffe train --solver=step1/solver_face.prototxt  --weights=step1/pretrain_iter_10000.caffemodel  --gpu 0 2>&1 | tee step1/log.log
