#include <vector>

#include "caffe/layers/quant_conv_layer.hpp"
#define SHIFT_MAX 13
#define EPSILON 1e-6

namespace caffe {

template <typename Dtype>
__global__ void ConvForward(const int nthreads, const Dtype* const bottom_data, const int num, 
        const int channels, const int height, const int width, const int conved_height,
        const int conved_width, const int kernel_h, const int kernel_w, const int stride_h, 
        const int stride_w, const int pad_h, const int pad_w, Dtype* const top_data, 
        const Dtype* const weight, const Dtype* const bias, const bool bias_term_) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int pw = index % conved_width;
        const int ph = (index / conved_width) % conved_height;
        const int c = (index / conved_width / conved_height) % channels;
        const int n = index / conved_width / conved_height / channels;
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);
        Dtype aveval = 0;
        const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
        const Dtype* const weight_slice = weight + c * kernel_h * kernel_w;
        int khstart = hend < kernel_h ? kernel_h - hend : 0;
        int kwstart = wend < kernel_w ? kernel_w - wend : 0;
        
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                aveval += bottom_slice[h * width + w] * weight_slice[(khstart + h - hstart) * kernel_w + (kwstart + w - wstart)];
            }
        }
        if(bias_term_) {
            aveval += bias[c];
        }
        top_data[index] = aveval;
    }
}

template <typename Dtype>
__global__ void ConvBackward(const int nthreads, const Dtype* const top_diff, const int num, 
        const int channels, const int height, const int width, const int conved_height, 
        const int conved_width, const int kernel_h, const int kernel_w, const int stride_h,
        const int stride_w, const int pad_h, const int pad_w, Dtype* const bottom_diff, 
        const Dtype* const weight) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int w = index % width + pad_w;
        const int h = (index / width) % height + pad_h;
        const int c = (index / width / height) % channels;
        const int n = index / width / height / channels;
        const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        const int phend = min(h / stride_h + 1, conved_height);
        const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        const int pwend = min(w / stride_w + 1, conved_width);
        const int khstart = (h >= kernel_h) ? ((h - kernel_h) % stride_h) + (kernel_h - stride_h) : h;
        const int kwstart = (w >= kernel_w) ? ((w - kernel_w) % stride_w) + (kernel_w - stride_w) : w;
        Dtype gradient = 0;
        const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
        const Dtype* const weight_slice = weight + c * kernel_h * kernel_w;
        
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int kh = khstart - (ph - phstart) * stride_h;
                int kw = kwstart - (pw - pwstart) * stride_w;
                gradient += top_diff_slice[ph * conved_width + pw] * weight_slice[kh * kernel_w + kw];
            }
        }
        bottom_diff[index] = gradient;
    }
}

__device__ float atomicAddme(float* address, float val) {
    return atomicAdd(address,val);
}

__device__ double atomicAddme(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#define DIVIDE_CEIL(a,b) a / b + ((a / b * b) < a)

template <typename Dtype>
__global__ void ConvBackwardWeight(const int nthreads, const Dtype* const top_diff,
        const int num, const int channels, const int height, const int width, 
        const int conved_height, const int conved_width, const int kernel_h, 
        const int kernel_w, const int stride_h, const int stride_w, const int pad_h, 
        const int pad_w, Dtype* const weight_diff, const Dtype* const bottom_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int kw = index % kernel_w;
        const int kh = (index / kernel_w) % kernel_h;
        const int c = index / kernel_w / kernel_h;
        Dtype gradient = 0;
        
        for (int n = 0; n < num; n++) {
            const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
            const Dtype* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;
            const int phstart = max(DIVIDE_CEIL((pad_h - kh), stride_h), 0);
            const int phend = min(DIVIDE_CEIL((height + pad_h - kh), stride_h), conved_height);
            const int pwstart = max(DIVIDE_CEIL((pad_w - kw), stride_w), 0);
            const int pwend = min(DIVIDE_CEIL((width + pad_w - kw), stride_w), conved_width);
        
            for (int ph = phstart; ph < phend; ph++) {
                for (int pw = pwstart; pw < pwend; pw++) {
                    const int h = ph * stride_h + kh - pad_h;
                    const int w = pw * stride_w + kw - pad_w;
                    gradient += top_diff_slice[ph * conved_width + pw] * bottom_data_slice[h * width + w];
                }
            }
        }
        weight_diff[c * kernel_h * kernel_w + kh * kernel_w + kw] += gradient;
    }
}

template <typename Dtype>
__global__ void ConvBackwardBias(const int nthreads, const Dtype* const top_diff,
        const int num, const int channels, const int height, const int width, 
        const int conved_height, const int conved_width, const int kernel_h, 
        const int kernel_w, const int stride_h, const int stride_w, const int pad_h, 
        const int pad_w, Dtype* const bias_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int c = index;
        Dtype gradient=0;
        for(int n = 0; n < num; n++) {
            const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
            for(int ph = 0; ph < conved_height; ph++) {
                for (int pw = 0; pw < conved_width; pw++) {
                    gradient += top_diff_slice[ph * conved_width + pw];
                }
            }
        }
        bias_diff[c] += gradient;
    }
}

// Dynamic fixed-point model: linearly quantize a block of data between {-max, +max} 
// Shift is determined by weight (bias is represented by 20-bit, no need to consider its shift)
template <typename Dtype>
void quantize_block_data_dynamic_fxp(Dtype *dst_weight, Dtype *src_weight, int count_weight, int bw_param, bool shift_enable, Dtype *dst_bias=NULL, Dtype *src_bias=NULL, int count_bias=0)
{   
    int num_steps = (1<<(bw_param-1)) - 1;   // half side quantize steps, +/-127 for 8-bits

    // 1: search abs max:
    double abs_data, abs_max_w;
    double grid, shift_decimal;
    abs_max_w = 0;
    for (int i = 0; i < count_weight; i++) {
	abs_data = fabs(src_weight[i]);
        abs_max_w = (abs_max_w < abs_data) ? abs_data : abs_max_w;
    }

    int shift_max_value = SHIFT_MAX;
    if (bw_param == 8) shift_max_value = 15;

    // 2: quantize weight and bias 
    if (shift_enable == true) {
       // Shift_based quantization 
       int shift_value = floor(std::log(double(num_steps) / abs_max_w) / std::log(2.0)); 
       shift_value = std::max(shift_value, 0); 
       shift_value = std::min(shift_value, shift_max_value); 
       shift_decimal = double(1 << shift_value); 

       for (int i = 0; i < count_weight; i++) {
           Dtype data = double(src_weight[i]);
           if (data >= 0)
               dst_weight[i] = floor(data * shift_decimal + 0.5) / shift_decimal;
           else
               dst_weight[i] = ceil(data * shift_decimal - 0.5) / shift_decimal;
       }
       if((dst_bias) && (src_bias)){
         for (int i = 0; i < count_bias; i ++) {
             Dtype data = double(src_bias[i]);
             if (data >= 0)
                 dst_bias[i] = floor(data * shift_decimal + 0.5) / shift_decimal;
             else
                 dst_bias[i] = ceil(data * shift_decimal - 0.5) / shift_decimal;
         }
       }
    }
    else {
       // Direct quantization using max_weight
       grid = double(abs_max_w/num_steps); 
       for (int i = 0; i < count_weight; i++) {
           //Dtype data = std::min(std::max(double(src_weight[i]), -abs_max_w), abs_max_w);
           Dtype data = double(src_weight[i]);
           if (data >= 0)
               dst_weight[i] = floor(data/grid  + 0.5) * grid;
           else
               dst_weight[i] = ceil(data/grid - 0.5) * grid;
       }
       // bias not quantized here
       if((dst_bias) && (src_bias)){
          for (int i = 0; i < count_bias; i ++) {
             dst_bias[i] = src_bias[i];
          }
       }
    }
} 

// Fixed-point model: quantize the data of the whole layer using 1/2/3/5-bit:    
template <typename Dtype>
void quantize_block_data_shift_quant(Dtype *dst_ptr, const Dtype *src_ptr, const int out_ch_num, const int in_ch_num, const int kernel_size, const int bw_param, bool shift_enable, const int quant_bit, Dtype *dst_bias=NULL, const Dtype * src_bias=NULL)
{
       int bw_bias = 20;
       if (bw_param == 12){
	  if (quant_bit == QuantConvolutionParameter_Precision_ONE_BIT) bw_bias = 12;
          else if (quant_bit == QuantConvolutionParameter_Precision_TWO_BITS) bw_bias = 12;
          else if (quant_bit == QuantConvolutionParameter_Precision_THREE_BITS) bw_bias = 18;
          else if (quant_bit == QuantConvolutionParameter_Precision_FIVE_BITS) bw_bias = 16;
       }
       // Step 1a: calculate max scalar/max weight/max bias per layer:
       double abs_max = 0;
       double abs_data;
       int offset = 0;
       for (int out_ch = 0; out_ch < out_ch_num; out_ch++) {
           for (int in_ch = 0; in_ch < in_ch_num; in_ch++) {
               for (int i = 0; i < kernel_size; i++) {
                   abs_data = fabs(*(src_ptr + offset + i));
                   abs_max = (abs_max < abs_data) ? abs_data : abs_max;
               }
               offset += kernel_size;
           }
       }

       // step 1b: calculate shift
       int shift_max_value = SHIFT_MAX;
       if (bw_param == 8) shift_max_value = 15;

       int shift_w, shift;  
       int shift_b = shift_max_value;  
       double shift_decimal;
       int num_steps = (1<<(bw_param-1)) - 1; 
       int num_steps_bias = (1<<(bw_bias-1)) - 1; 
       shift_w = floor(std::log(double(num_steps)/abs_max)/std::log(2.0));  
       
       if((dst_bias) && (src_bias)){
          double abs_max_bias = 0; 
          for (int i = 0; i < out_ch_num; i ++) {
              abs_data = double(fabs(src_bias[i]));
              abs_max_bias = (abs_max_bias < abs_data) ? abs_data : abs_max_bias;
          }
          shift_b = floor(std::log(double(num_steps_bias)/(abs_max_bias+EPSILON))/std::log(2.0));  
       }
       shift = std::min(shift_w, shift_b);
       shift = std::min(std::max(shift, 0), shift_max_value);
       shift_decimal = double(1 << shift); 

       double grid_scalar, quant_scalar, var;
       Dtype coef, qcoef;
       if (quant_bit == QuantConvolutionParameter_Precision_ONE_BIT) {

          // step 2: use shift to quantize scalar 
          offset = 0;
          for (int out_ch = 0; out_ch < out_ch_num; out_ch++) {
              for (int in_ch = 0; in_ch < in_ch_num; in_ch++) {
                  var = 0; 
                  for (int i = 0; i< kernel_size; i++){
                      abs_data = fabs(*(src_ptr + offset + i));
                      var += abs_data;
                  }
                  var /= kernel_size;
                  grid_scalar = (var + EPSILON)/1.f;
                  if(shift_enable == true) 
                    quant_scalar = floor(grid_scalar*shift_decimal + 0.5) / shift_decimal;
                  else
                    quant_scalar = grid_scalar;
                  for (int i = 0; i < kernel_size; i++) {
                      coef = *(src_ptr + offset + i);
       	              qcoef = (coef >= 0) ? quant_scalar : -quant_scalar;
                      *(dst_ptr + offset + i) = qcoef;
                  }
                  offset += kernel_size;
              } 
          }
       }
       else if (quant_bit == QuantConvolutionParameter_Precision_TWO_BITS) {

          // step 2: use shift to quantize scalar 
          offset = 0;
          for (int out_ch = 0; out_ch < out_ch_num; out_ch++) {
              for (int in_ch = 0; in_ch < in_ch_num; in_ch++) {
                  var = 0; 
                  for (int i = 0; i< kernel_size; i++){
                      abs_data = fabs(*(src_ptr + offset + i));
                      var += abs_data;
                  }
                  grid_scalar = var/kernel_size;
                  if(shift_enable == true) 
                    quant_scalar = floor(grid_scalar*shift_decimal + 0.5) / shift_decimal;
                  else
                    quant_scalar = grid_scalar;
                  
                  for (int i = 0; i < kernel_size; i++) {
                      coef = *(src_ptr + offset + i);
       	              qcoef = (fabs(coef) < 0.25*quant_scalar) ? 0.0 : (coef >= 0) ? quant_scalar : -quant_scalar;
                      *(dst_ptr +offset +  i) = qcoef;
                  }
                  offset += kernel_size;
              } 
          }
       } 
       else if (quant_bit == QuantConvolutionParameter_Precision_THREE_BITS) {
          Dtype abs_coef;
          
          // step 2: use shift to quantize scalar 
          offset = 0;
          for (int out_ch = 0; out_ch < out_ch_num; out_ch++) {
              for (int in_ch = 0; in_ch < in_ch_num; in_ch++) {
                  
                  var = 0; 
                  for (int i = 0; i< kernel_size; i++){
                      abs_data = fabs(*(src_ptr + offset + i));
                      var += abs_data;
                  }
                  var /= kernel_size;
                  grid_scalar = (var + EPSILON)/4.f;
                  if(shift_enable == true) 
                    quant_scalar = floor(grid_scalar*shift_decimal + 0.5) / shift_decimal;
                  else
                    quant_scalar = grid_scalar;

                  for (int i = 0; i < kernel_size; i++) {
                      coef = *(src_ptr + offset + i);
                      int abs_coefInt = fabs(coef)/grid_scalar;

                      abs_coefInt = (abs_coefInt < 3) ? abs_coefInt : 4;
                      abs_coef = abs_coefInt*quant_scalar;
                      
                      qcoef = (coef >= 0) ? abs_coef : -abs_coef;

                      *(dst_ptr + offset + i) = qcoef;
                  }
                  offset += kernel_size;
              } 
          }
       }

       else if (quant_bit == QuantConvolutionParameter_Precision_FIVE_BITS) {
         
          // step 2: use shift to quantize scalar 
          int n_bit = 5; 

          offset = 0;
          for (int out_ch = 0; out_ch < out_ch_num; out_ch++) {
              for (int in_ch = 0; in_ch < in_ch_num; in_ch++) {
                  
                  var = 0; 
                  for (int i = 0; i< kernel_size; i++){
                      abs_data = fabs(*(src_ptr + offset + i));
                      var = (var < abs_data) ? abs_data : var;
                  }
                  grid_scalar = (var + EPSILON)/double((1 << (n_bit - 1)) - 1);
                  if(shift_enable == true) 
                    quant_scalar = floor(grid_scalar*shift_decimal + 0.5) / shift_decimal; 
                  else
                    quant_scalar = grid_scalar;

                  for (int i = 0; i < kernel_size; i++) {
                      coef = *(src_ptr + offset +i);
                      if (coef >= 0)
                         qcoef = floor(coef/grid_scalar + 0.5)*quant_scalar;
                      else
                         qcoef = ceil(coef/grid_scalar - 0.5)*quant_scalar;

                      *(dst_ptr + offset + i) = qcoef;

                  }
                  offset += kernel_size;
              } 
          }
       } 
       else {
          printf("Error QuantConvolutionParameter: Precision\n");
          exit(0);
       }

      if((dst_bias) && (src_bias)){
        if(shift_enable == true) 
           for (int i = 0; i < out_ch_num; i ++) {
               Dtype data = double(src_bias[i]);
               if (data >= 0)
                   dst_bias[i] = floor(data * shift_decimal + 0.5) / shift_decimal;
               else
                   dst_bias[i] = ceil(data * shift_decimal - 0.5) / shift_decimal;
           }
        else 
           // bias not quantized here
           for (int i = 0; i < out_ch_num; i ++) {
               dst_bias[i] = src_bias[i];
           }
      }     
}

template <typename Dtype>
void QuantConvolutionLayer<Dtype>::forward_conv_gpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top, const Dtype* weight, const Dtype* bias) {
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
            if (this->bias_term_) {
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
}

template<typename Dtype>
void QuantConvolutionLayer<Dtype>::forward_dw_conv_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top, const Dtype* weight, const Dtype* bias) {
    int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
    int* stride_data = this->stride_.mutable_cpu_data();
    int* pad_data = this->pad_.mutable_cpu_data();
    
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        const int count = top[i]->count();
        vector<int> shape_ = bottom[i]->shape();
        const int channels_ = shape_[1];
        const int height_ = shape_[2];
        const int width_ = shape_[3];
        const int kernel_h_ = kernel_shape_data[0];
        const int kernel_w_ = kernel_shape_data[1];
        const int stride_h_ = stride_data[0];
        const int stride_w_ = stride_data[1];
        const int pad_h_ = pad_data[0];
        const int pad_w_ = pad_data[1];
        const int conved_height = this->output_shape_[0];
        const int conved_weight = this->output_shape_[1];
        const bool bias_term_ = this->bias_term_;
        
        if (bias_term_) {
            ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, bottom[i]->num(), channels_,
                    height_, width_,conved_height,conved_weight,kernel_h_,
                    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
        } else {
            ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, bottom[i]->num(), channels_,
                    height_, width_,conved_height,conved_weight,kernel_h_,
                    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
        }
    }
}

template<typename Dtype>
void QuantConvolutionLayer<Dtype>::backward_conv_gpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, const Dtype* weight) {
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        // Bias gradient, if necessary.
        if (this->bias_term_ && this->param_propagate_down_[1]) {
            Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
            for (int n = 0; n < this->num_; ++n) {
                this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
            }
        }
        if (this->param_propagate_down_[0] || propagate_down[i]) {
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            for (int n = 0; n < this->num_; ++n) {
                // gradient w.r.t. weight. Note that we will accumulate diffs.
                if (this->param_propagate_down_[0]) {
                    this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                            top_diff + n * this->top_dim_, weight_diff);
                }
                // gradient w.r.t. bottom data, if necessary.
                if (propagate_down[i]) {
                    this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                            bottom_diff + n * this->bottom_dim_);
                }
            }
        }
    }
}

template<typename Dtype>
void QuantConvolutionLayer<Dtype>::backward_dw_conv_gpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, const Dtype* weight) {
    int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
    int* stride_data = this->stride_.mutable_cpu_data();
    int* pad_data = this->pad_.mutable_cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const bool bias_term_ = this->bias_term_;
    Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
    const bool bias_propagate_down_ = this->param_propagate_down_[1];
    const bool weight_propagate_down_ = this->param_propagate_down_[0];
    const int kernel_h_ = kernel_shape_data[0];
    const int kernel_w_ = kernel_shape_data[1];
    const int stride_h_ = stride_data[0];
    const int stride_w_ = stride_data[1];
    const int pad_h_ = pad_data[0];
    const int pad_w_ = pad_data[1];
    const int conved_height = this->output_shape_[0];
    const int conved_weight = this->output_shape_[1];
    
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        vector<int> shape_ = bottom[i]->shape();
        const int channels_ = shape_[1];
        const int height_ = shape_[2];
        const int width_ = shape_[3];

        // Bias gradient, if necessary.
        if (bias_term_ && bias_propagate_down_) {
            const int count_bias = channels_;
            ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS>>>(
                    count_bias, top_diff, bottom[i]->num(), channels_, height_, width_, conved_height,
                    conved_weight, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bias_diff);
        }
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (weight_propagate_down_) {
            const int count_weight = channels_ * kernel_h_ * kernel_w_;
            ConvBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>(
                    count_weight, top_diff, bottom[i]->num(), channels_, height_, width_, conved_height, 
                    conved_weight, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, weight_diff, bottom_data);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
            const int count_bottom=bottom[i]->count();
            ConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
                    count_bottom, top_diff, bottom[i]->num(), channels_, height_, width_, conved_height, conved_weight,
                    kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff, weight);
        }
    }
}

template <typename Dtype>
void QuantConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//quantize coef and run forward process:
	int CoefPrecision = this->layer_param_.quant_convolution_param().coef_precision();
        bool shift_enable =  this->layer_param_.quant_convolution_param().shift_enable();
        int bw_param = this->layer_param_.quant_convolution_param().bw_params();
        int num_output = this->layer_param_.convolution_param().num_output();
        int group = this->layer_param_.convolution_param().group();
        bool dw = num_output == group;


	if (CoefPrecision == QuantConvolutionParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:
          	const Dtype* weight = this->blobs_[0]->gpu_data();
        	const Dtype* bias = (this->bias_term_) ? this->blobs_[1]->gpu_data() : 0;
        	dw ? forward_dw_conv_gpu(bottom, top, weight, bias) : forward_conv_gpu(bottom, top, weight, bias);                     
	}
	else if ((CoefPrecision == QuantConvolutionParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_TWO_BITS)||    // two bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_THREE_BITS)|| // three bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_FIVE_BITS)){   // five bits coef scheme:
		// coef quantization: one or two bits
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);
                int Kernel_Size = Kernel_Height*Kernel_Width;

		const Dtype* pt_origin_coef = this->blobs_[0]->cpu_data();	// in cpu
		Dtype* pt_binary_coef = blob_binary_coef.mutable_cpu_data();

                // weight quantization with a floating scalar
                Blob<Dtype> blob_binary_bias;
                if (this->bias_term_){
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   const Dtype* pt_origin_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* pt_binary_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
                   quantize_block_data_shift_quant(pt_binary_coef, pt_origin_coef, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision, pt_binary_bias, pt_origin_bias);
                }
                else {
                   quantize_block_data_shift_quant(pt_binary_coef, pt_origin_coef, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision);
                }
		// run convolution:
		const Dtype* weight = blob_binary_coef.gpu_data();
        	const Dtype* bias = (this->bias_term_) ? blob_binary_bias.gpu_data() : 0;
        	dw ? forward_dw_conv_gpu(bottom, top, weight, bias) : forward_conv_gpu(bottom, top, weight, bias);    
	}
	else if (CoefPrecision == QuantConvolutionParameter_Precision_DYNAMIC_FIXED_POINT) {
   		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

    	        Dtype* src_weight = (Dtype*) this->blobs_[0]->cpu_data();
	        Dtype* dst_weight = (Dtype*)blob_binary_coef.mutable_cpu_data();
                int count_weight = Output_Channels*Input_Channels*Kernel_Height*Kernel_Width;

                Blob<Dtype> blob_binary_bias;
                if (this->bias_term_){
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   Dtype* src_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* dst_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
        	   int count_bias = Output_Channels;
                   quantize_block_data_dynamic_fxp(dst_weight, src_weight, count_weight, bw_param, shift_enable, dst_bias, src_bias, count_bias);
                }
                else {
                   quantize_block_data_dynamic_fxp(dst_weight, src_weight, count_weight, bw_param, shift_enable);
                }

		// run convolution:
		const Dtype* weight = blob_binary_coef.gpu_data();
        	const Dtype* bias = (this->bias_term_) ? blob_binary_bias.gpu_data() : 0;
        	dw ? forward_dw_conv_gpu(bottom, top, weight, bias) : forward_conv_gpu(bottom, top, weight, bias);
	}
	else {
		printf("Error quantization scheme !!!\n");
	}
}

template <typename Dtype>
void QuantConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int CoefPrecision = this->layer_param_.quant_convolution_param().coef_precision();
        bool shift_enable =  this->layer_param_.quant_convolution_param().shift_enable();
        int bw_param = this->layer_param_.quant_convolution_param().bw_params();
        int num_output = this->layer_param_.convolution_param().num_output();
        int group = this->layer_param_.convolution_param().group();
        bool dw = num_output == group;

	if (CoefPrecision == QuantConvolutionParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:
		const Dtype* weight = this->blobs_[0]->gpu_data();
	        dw ? backward_dw_conv_gpu(top, propagate_down, bottom, weight) : backward_conv_gpu(top, propagate_down, bottom, weight);
	}
	else if ((CoefPrecision == QuantConvolutionParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_TWO_BITS)||    // two bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_THREE_BITS)||  // three bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_FIVE_BITS)) {  // five bits coef scheme:
		//std::cout << "======= Binary_Backward_cpu ============\n";

		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);
                int Kernel_Size = Kernel_Height*Kernel_Width;

		const Dtype* pt_origin_coef = this->blobs_[0]->cpu_data();	// in cpu
		Dtype* pt_binary_coef = blob_binary_coef.mutable_cpu_data();

                if (this->bias_term_){
                   Blob<Dtype> blob_binary_bias;
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   const Dtype* pt_origin_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* pt_binary_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
                   quantize_block_data_shift_quant(pt_binary_coef, pt_origin_coef, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision, pt_binary_bias, pt_origin_bias);
                }
                else {
                   quantize_block_data_shift_quant(pt_binary_coef, pt_origin_coef, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision);
                }
        	const Dtype* weight = blob_binary_coef.gpu_data();
        	dw ? backward_dw_conv_gpu(top, propagate_down, bottom, weight) : backward_conv_gpu(top, propagate_down, bottom, weight);
	}
	else if (CoefPrecision == QuantConvolutionParameter_Precision_DYNAMIC_FIXED_POINT) {
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);
   
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

    	        Dtype* src_weight = (Dtype*) this->blobs_[0]->cpu_data();
	        Dtype* dst_weight = (Dtype*) blob_binary_coef.mutable_cpu_data();
                int count_weight = Output_Channels*Input_Channels*Kernel_Height*Kernel_Width;

                if (this->bias_term_){
                   Blob<Dtype> blob_binary_bias;
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   Dtype* src_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* dst_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
        	   int count_bias = Output_Channels;
                   quantize_block_data_dynamic_fxp(dst_weight, src_weight, count_weight, bw_param, shift_enable, dst_bias, src_bias, count_bias);
                }
                else {
                   quantize_block_data_dynamic_fxp(dst_weight, src_weight, count_weight, bw_param, shift_enable);
                }

                // backward process
		const Dtype* weight = blob_binary_coef.gpu_data();
        	dw ? backward_dw_conv_gpu(top, propagate_down, bottom, weight) : backward_conv_gpu(top, propagate_down, bottom, weight);
	}
	else {
		printf("Error quantization scheme !!!\n");
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantConvolutionLayer);

//need these lines to instantiate static functions from hpp that are not precisely forward_gpu and backward_gpu
//as instantiate_layer_gpu_funcs does this under the hood.

//regular forward conv
template void QuantConvolutionLayer<float>::forward_conv_gpu(const vector<Blob<float>*>& bottom, 
        const vector<Blob<float>*>& top, const float* weight, const float* bias);
template void QuantConvolutionLayer<double>::forward_conv_gpu(const vector<Blob<double>*>& bottom, 
        const vector<Blob<double>*>& top, const double* weight, const double* bias);
    
//dw forward conv
template void QuantConvolutionLayer<float>::forward_dw_conv_gpu(const vector<Blob<float>*>& bottom, 
        const vector<Blob<float>*>& top, const float* weight, const float* bias);
template void QuantConvolutionLayer<double>::forward_dw_conv_gpu(const vector<Blob<double>*>& bottom, 
        const vector<Blob<double>*>& top, const double* weight, const double* bias);
    
//regular backward conv
template void QuantConvolutionLayer<float>::backward_conv_gpu(
        const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<float>*>& bottom, const float* weight);
template void QuantConvolutionLayer<double>::backward_conv_gpu(
        const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<double>*>& bottom, const double* weight);
    
//dw backward conv
template void QuantConvolutionLayer<float>::backward_dw_conv_gpu(
        const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<float>*>& bottom, const float* weight);
template void QuantConvolutionLayer<double>::backward_dw_conv_gpu(
        const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<double>*>& bottom, const double* weight);

}  // namespace caffe
