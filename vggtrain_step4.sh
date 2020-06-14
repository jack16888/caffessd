#!/bin/sh
build/tools/caffe train --solver=step4/solver_face.prototxt --weights=step4/net_refined.caffemodel --gpu 0 2>&1 | tee step4/log.log
