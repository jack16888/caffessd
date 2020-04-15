#include <vector>

#include "caffe/layers/quant_conv_layer.hpp"
#define SHIFT_MAX 13
#define EPSILON 1e-6

namespace caffe {

template <typename Dtype>
void QuantConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// Dynamic fixed-point model: linearly quantize a block of data between {-max, +max} 
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

       // step 1: calculate shift
       //shift_w = floor(std::log(double(num_steps)/abs_max)/std::log(2.0));  
       //shift_w = std::min(std::max(shift_w, 0), SHIFT_MAX);
       //shift_decimal = double(1 << shift_w); 

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
void QuantConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        printf("svic foward cpu\n");
	//quantize coef and run forward process:
	int CoefPrecision = this->layer_param_.quant_convolution_param().coef_precision();
        bool shift_enable =  this->layer_param_.quant_convolution_param().shift_enable();
        int bw_param = this->layer_param_.quant_convolution_param().bw_params();

	if (CoefPrecision == QuantConvolutionParameter_Precision_FLOATING_POINT) {// floating point coef scheme:
		// conv layer coef: no coef conversion.
		const Dtype* weight = this->blobs_[0]->cpu_data();

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
	else if ((CoefPrecision == QuantConvolutionParameter_Precision_ONE_BIT)   ||    // one bit coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_TWO_BITS)  ||    // two bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_THREE_BITS)|| // three bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_FIVE_BITS))    // five bits coef scheme:
        {  
		// coef quantization: one bit
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);
		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);
                int Kernel_Size = Kernel_Height*Kernel_Width;

		const Dtype* src_weight = (Dtype*) this->blobs_[0]->cpu_data();
		Dtype* dst_weight = (Dtype*)blob_binary_coef.cpu_data();

                Blob<Dtype> blob_binary_bias;
                // weight quantization with a floating scalar
                if (this->bias_term_){
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   const Dtype* src_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* dst_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
                   quantize_block_data_shift_quant(dst_weight, src_weight, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision, dst_bias, src_bias);
                }
                else {
                   quantize_block_data_shift_quant(dst_weight, src_weight, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision);
                }

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, dst_weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					//const Dtype* bias = this->blobs_[1]->cpu_data();
		                        const Dtype* bias = blob_binary_bias.cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
					//this->forward_cpu_bias(top_data + n * this->top_dim_, dst_bias);
				}
			}
		}

	}// end of 1/2/3/5 bit

	else if (CoefPrecision ==QuantConvolutionParameter_Precision_DYNAMIC_FIXED_POINT) {

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
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
		        const Dtype* weight = blob_binary_coef.cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
		    		        //const Dtype* bias = this->blobs_[1]->cpu_data();
		                        const Dtype* bias = blob_binary_bias.cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}// end of fixed point

	else {
		printf("Error quantization scheme !!!\n");
	}

}

template <typename Dtype>
void QuantConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        printf("svic backward cpu\n");

	int CoefPrecision = this->layer_param_.quant_convolution_param().coef_precision();
        bool shift_enable =  this->layer_param_.quant_convolution_param().shift_enable();
        int bw_param = this->layer_param_.quant_convolution_param().bw_params();

	if (CoefPrecision == QuantConvolutionParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:

		const Dtype* weight = this->blobs_[0]->cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else if ((CoefPrecision == QuantConvolutionParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_TWO_BITS)||	  // two bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_THREE_BITS) || // three bits coef scheme:
	         (CoefPrecision == QuantConvolutionParameter_Precision_FIVE_BITS)) {  // five bits coef scheme:

		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);
		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);
                int Kernel_Size = Kernel_Height*Kernel_Width;

		const Dtype* src_weight = this->blobs_[0]->cpu_data();
		Dtype* dst_weight = blob_binary_coef.mutable_cpu_data();

                // weight quantization with a floating scalar
                if (this->bias_term_){
                   Blob<Dtype> blob_binary_bias;
		   blob_binary_bias.ReshapeLike(*this->blobs_[1]);
    		   const Dtype* src_bias = (Dtype*) this->blobs_[1]->cpu_data();
	    	   Dtype* dst_bias = (Dtype*)blob_binary_bias.mutable_cpu_data();
                   quantize_block_data_shift_quant(dst_weight, src_weight, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision, dst_bias, src_bias);
                }
                else {
                   quantize_block_data_shift_quant(dst_weight, src_weight, Output_Channels, Input_Channels, Kernel_Size, bw_param, shift_enable, CoefPrecision);
                }

		// run backward prop:
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, dst_weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}

	}
	else if (CoefPrecision == QuantConvolutionParameter_Precision_DYNAMIC_FIXED_POINT) {
		// coef quantization: Dynamic Fixed Point
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

    	        Dtype* src_weight = (Dtype*) this->blobs_[0]->cpu_data();
	        Dtype* dst_weight = (Dtype*)blob_binary_coef.mutable_cpu_data();
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
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		for (int i=0; i<top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
			    const Dtype* bottom_data = bottom[i]->cpu_data();
			    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		            const Dtype* weight = blob_binary_coef.cpu_data();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else {
		printf("Error quantization scheme !!!\n");
	}
}

#ifdef CPU_ONLY
STUB_GPU(QuantConvolutionLayer);
#endif

INSTANTIATE_CLASS(QuantConvolutionLayer);
REGISTER_LAYER_CLASS(QuantConvolution);
}  // namespace caffe
