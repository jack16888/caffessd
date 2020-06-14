#include <algorithm>
#include <vector>
#include <math.h>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/quant_relu_layer.hpp"

namespace caffe {

// CUDA kernele for forward

template <typename Dtype>
__global__ void QuantReLUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* alpha_data,
    const int div_factor, const int qsteps, float negative_slope, bool quant_enable) {
    CUDA_KERNEL_LOOP(index, n) {
    if (quant_enable){ 
       int c = (index / dim) % channels / div_factor;
       Dtype d_clamp = (in[index] < 0) ? 0 : (in[index] <= alpha_data[c]) ? in[index] : alpha_data[c];
       int d_quant = (int)(d_clamp * (Dtype)qsteps / alpha_data[c] + 0.5f);
       out[index] = (Dtype)d_quant * alpha_data[c] / (Dtype)qsteps;
    }
    else {
       out[index] = max(in[index], Dtype(0))
             + negative_slope * min(in[index], Dtype(0));
    }
  }
}

// CUDA kernel for bottom backward

template <typename Dtype>
__global__ void QuantReLUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* alpha_data, const int div_factor, float negative_slope, bool quant_enable) {
    CUDA_KERNEL_LOOP(index, n) {
    if (quant_enable) {
       int c = (index / dim) % channels / div_factor;
       out_diff[index] = in_diff[index] * ((in_data[index] < alpha_data[c]) && (in_data[index] > 0));
    }
    else {
       out_diff[index] = in_diff[index] * ((in_data[index] > 0)
               + negative_slope * (in_data[index] <= 0));
    }
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void QuantReLUParamBackward(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const Dtype* alpha_data, const int div_factor, bool quant_enable) {
    CUDA_KERNEL_LOOP(index, n) {
    if (quant_enable){
       int c = (index / dim) % channels / div_factor;
       out_diff[index] = in_diff[index] * (in_data[index] >= alpha_data[c]);
       for ( int k = 1; k < rows; k++ ) {
           int ii = index + k*rowPitch;
           out_diff[index] += in_diff[ii] * (in_data[ii] >= alpha_data[c]);
       }
    }
    else {
       out_diff[index] = Dtype(0);
    }
  }
}

template <typename Dtype>
void QuantReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }
  
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* alpha_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;
  const int qsteps = (1<<act_bits_) - 1;
  // NOLINT_NEXT_LINE(whitespace/operators)
  QuantReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
     count, channels, dim, bottom_data, top_data, alpha_data, div_factor, qsteps, negative_slope_, quant_enable_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void QuantReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0] && quant_enable_) {
    const Dtype* alpha_data = this->blobs_[0]->gpu_data();
    Dtype* alpha_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;
    const int div_factor = channel_shared_ ? channels : 1;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
      QuantReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
        CAFFE_CUDA_NUM_THREADS>>>(
        cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), top_diff,
        bottom_data,
        backward_buff_.mutable_gpu_diff(), alpha_data, div_factor, quant_enable_);
      CUDA_POST_KERNEL_CHECK;


    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), alpha_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        alpha_diff);
    }
  }

  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
       const Dtype* alpha_data = this->blobs_[0]->gpu_data();
       int div_factor = channel_shared_ ? channels : 1;
       // NOLINT_NEXT_LINE(whitespace/operators)
 
       QuantReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
         CAFFE_CUDA_NUM_THREADS>>>(
         count, channels, dim, top_diff, bottom_data, bottom_diff, alpha_data,
         div_factor, negative_slope_, quant_enable_);
       CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(QuantReLULayer);


}  // namespace caffe
