# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer


class QOlc(nn.Module):

    def __init__(self, in_chans, quant=False):
        super(QOlc, self).__init__()

        self.quant = quant
        self.mode = 'pass'
        self.in_chans = in_chans
        
        self.scale = torch.ones((1, self.in_chans, 1, 1)).to(device='cuda:0')
        self.add = torch.zeros((1, self.in_chans, 1, 1)).to(device='cuda:0')
        self.ptf = None
        self.qkv_scale = torch.ones((3, 1, 1, 1, 1)).to(device='cuda:0')

    def forward(self,
                x,
                quantizer=None):
        # print(x.shape, len(x.shape))
        if self.mode == 'pass':
            if len(x.shape) == 4:
                x = (x - self.add) / self.scale
            elif len(x.shape) == 5: ## qkv
                x = x / self.qkv_scale
            else:
                self.scale = self.scale.view(1, 1, self.in_chans)
                x = (x - self.add) / self.scale

            return x

        elif self.mode == 'scaling':
            if len(x.shape) == 4:
                x = (x - self.add) / self.scale
            else:
                self.scale = self.scale.view(1, 1, self.in_chans)
                x = (x - self.add) / self.scale

            return x


class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.bits = bit_type.bits

    def quantize(self, w, n_bits=8):
    
        # Tensor-wise & Symmetric Quant for weights
        # (out_c, in_c, k, k)
        scales = w.abs().view(w.size(0), -1).max(dim=1).values
        #print(scales.max())
        q_max = 2**(n_bits-1)-1
        scales_reshaped = scales.view(-1, 1, 1, 1)

        scales_reshaped.clamp_(min=1e-5).div_(q_max)
        #w.clamp(-15, 15)
        w.div_(scales_reshaped).round_().mul_(scales_reshaped)

        return w

    def forward(self, x):
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


        weight = self.quantize(self.weight, self.bits)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)

class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 olc_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform',
                 FC2=False):
        super(QAct, self).__init__()
        self.FC2=FC2

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.olc_calibrate = olc_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        self.olc_observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.olc_quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.olc_observer, self.module_type)
    def forward(self, x):        
        
        if self.olc_calibrate:
            self.olc_quantizer.observer.update(x)
        
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            
            return x
        if self.FC2==True:
            pass
        else:
            x = self.quantizer(x)
            return x


class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 8
        N = torch.clamp(bit - 1 - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2 ** bit - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        if self.mode == 'ln':
                x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            
            #print(in_scale.shape, scale.shape)
            
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask
            
            #TODO: Verification Original Version(NJ)
            #mean_x_q = x_q.mean(dim=-1) * (in_scale1 /channel_nums)
            mean_x_q = x_q.mean(dim=-1) * in_scale1
            
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / std_x_q).unsqueeze(-1) *
                  self.weight.reshape(1, 1, -1)) / out_scale *
                torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x

class QIntSoftmax(nn.Module):

    def __init__(self,
                log_i_softmax=False,
                quant=False,
                calibrate=False,
                last_calibrate=False,
                bit_type=BIT_TYPE_DICT['int8'],
                calibration_mode='layer_wise',
                observer_str='minmax',
                quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                        self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                        self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):
        
        def int_linear(x_int, scaling_factor):
            coef = [0.84342419, 0.99126546]
            #coef = [0.8434, 0.9912]
            #coef = [0.845111188584348, 1.0]
            coef[1] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor) # Alg 8
            z = x_int + b_int # Alg 10
            #z = x_int * z     # Alg 10
            scaling_factor = coef[0] * scaling_factor # Alg 9
            return z, scaling_factor

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor) # Alg 8
            c_int = torch.floor(coef[2] / scaling_factor**2) # Alg 8
            z = x_int + b_int # Alg 10
            z = x_int * z     # Alg 10
            z = z + c_int     # Alg 10
            scaling_factor = coef[0] * scaling_factor**2 # Alg 9
            return z, scaling_factor

        def int_poly_8(x_int, scaling_factor):
            coef = [0.08947606, 0.12121466]
            coef[1] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor) # Alg 8
            z = x_int + b_int # Alg 10
            scaling_factor = coef[0] * scaling_factor # Alg 9
            #z = z/8
            
            return z, scaling_factor

        def int_exp_first_8(x_int, scaling_factor):

            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor) # Alg 4
                        
            x_int = torch.max(x_int, n * x0_int) # Alg 5
            q = torch.floor(x_int / x0_int)# Alg 6            
            r = x_int - x0_int * q # Alg 7
            exp_int, exp_scaling_factor = int_poly_8(r, scaling_factor) # Alg 8~10

            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0) # Alg 11
            
            scaling_factor = exp_scaling_factor / 2**n # Alg 11
            return exp_int, scaling_factor 


        ## Add Round 
        x_int = (x / scaling_factor).round()
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, exp_scaling_factor = int_exp_first_8(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)        
        
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
                        
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax

        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:          
                return x
            x = self.quantizer(x)
        
            return x