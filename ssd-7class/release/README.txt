# Caffe training and conversion tool

Usage:

1. Update your caffe installation by following the instructions in caffe-installation.

2. Train a valid model in caffe - example code is provided in training_example. We recommend using the sample script train.sh.

3. Go to conversion_tool/ 
   Run the conversion command:
```
python convert.py \
    *.prototxt \
    *.caffemodel \
    network*.json \
    fullmodel_def*.json \
    -o
    [--label=]{path} \
    [--shift_max=]{INT} \
    [--input_scale=]{FLOAT} \
    [--ouput_dir=]{path} \
    [--evaluate=]{path} \
    [--debug=]{true|false} \
    [--edit_gain=]{true|false}
```
Also see `python convert.py -h` for more usage instructions.

4. Use sample code in SDK.

