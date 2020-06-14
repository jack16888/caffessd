#!/bin/sh
build/tools/caffe train --solver=mobilessd_step1/solver_face.prototxt --weights=mobilessd_step1/step1_iter_30000.caffemodel  --gpu 0 2>&1 | tee mobilessd_step1/log.log
#build/tools/caffe train --solver=step1/solver_face.prototxt --weights=step1/pretrain_iter_200000.caffemodel --gpu 0 2>&1 | tee step1/log.log
#build/tools/caffe train --solver=mobilessd_step1/solver_face.prototxt  --weights=step1/pretrain_iter_105000.caffemodel  --gpu 0 2>&1 | tee step1/log.log
#build/tools/caffe train --solver=step1/solver_face.prototxt  --weights=step1/pretrain_iter_10000.caffemodel  --gpu 0 2>&1 | tee step1/log.log
