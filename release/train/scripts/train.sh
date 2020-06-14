/home/user/Apps/caffe/build/tools/caffe train \
--solver=solver.prototxt \
--weights=init.caffemodel \
--gpu all 2>&1 | tee log.log
