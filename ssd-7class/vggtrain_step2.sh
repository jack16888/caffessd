#!/bin/sh
build/tools/caffe train --solver=step2/solver_face.prototxt --weights=step2/step1_iter_170000.caffemodel --gpu 0 2>&1 | tee step2/log.log
