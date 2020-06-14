#!/bin/sh
build/tools/caffe train --solver=step3/solver_face.prototxt --weights=step3/step2_iter_15000_QuantReLU.caffemodel --gpu 0 2>&1 | tee step3/log.log
