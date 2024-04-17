import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from timm.layers import ConvNormAct
from models.layers.conv_bn_act import ConvNormAct
from torch.nn import functional as F
from models.byobnet import BottleneckBlock
from models.mobilevit import MobileVitBlock
#from timm.models.mobilevit import MobileVitBlock, MobileVitV2Block
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch import optim

from models.ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear, QOlc



def fold(m):
    bn_weight = m.bn.weight
    bn_bias = m.bn.bias
    bn_mean = m.bn.running_mean
    bn_var_sqrt = torch.sqrt(m.bn.running_var + m.bn.eps)

    scale_factor = bn_weight / bn_var_sqrt
    m.conv.weight.data = m.conv.weight * scale_factor.view(-1, 1, 1, 1)

    if m.conv.bias is not None:
        m.conv.bias.data = (m.conv.bias - bn_mean) * scale_factor + bn_bias
    else:
        m.bn.bias.data = bn_bias - bn_mean * scale_factor

    # After BN folding, reset bn parameters
    m.bn.weight.data.fill_(1.0)
    m.bn.running_mean.data.zero_()
    m.bn.running_var.data.fill_(1.0)

def bn_fold(model):
    print("BN Folding...")
    os.makedirs('weight_dist', exist_ok=True)  # create directory if not exist

    for name, module in model.named_modules():

        if isinstance(module, ConvNormAct):
            fold(module)





def quantize(w, n_bits=8):
    
    
    # Tensor-wise & Symmetric Quant for weights
    scales = w.abs().view(w.size(0), -1).max(dim=1).values
    q_max = 2**(n_bits-1)-1
    scales_reshaped = scales.view(-1, 1, 1, 1)

    scales_reshaped = scales_reshaped.clamp(min=1e-5).div(q_max)
    w = w.div(scales_reshaped).round().mul(scales_reshaped)

    return w

def act_quantize(inputs, bits=8):
    eps = 1.1920928955078125e-07
    qmax = 255
    qmin = 0
    
    max_val = torch.max(inputs)
    min_val = torch.min(inputs)
    
    scale = (max_val - min_val) / float(qmax - qmin)
    scale.clamp_(eps)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point.clamp_(qmin, qmax)    
    
    ## Quant
    outputs = inputs/ scale + zero_point
    outputs = outputs.round().clamp(qmin, qmax)


    ## Dequant
    outputs = (outputs - zero_point) * scale
    
    return outputs
  
        
def quant_loss(act, conv, iterations=1000, lr=1e-2):
    # Define alpha as a trainable parameter
    alpha_shape = (1, act.input.shape[1], 1, 1)
    alpha = nn.Parameter(torch.ones(alpha_shape, device=act.input.device)) 


    optimizer = optim.Adam([alpha], lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    act.input.requires_grad_()

    # Initialize variables to store the minimum loss and best alpha
    min_loss = float('inf')
    best_alpha = None

    bits = 8
    
    
    with torch.enable_grad():
        for i in range(iterations):
            optimizer.zero_grad()

            # if i % 200 == 0 and bits > 8:
            #     bits = bits - 2
            #     print(bits)

            y = F.conv2d(act.input, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


            if conv.groups == 1:
                smoothed_w = conv.weight * alpha
            else:
                smoothed_w = conv.weight * alpha.reshape(-1, 1, 1, 1)
            smoothed_x = act.input / alpha

            x_q = act_quantize(smoothed_x, bits)
            w_q = quantize(smoothed_w, bits)

            y_q = F.conv2d(x_q, w_q, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

            loss = criterion(y, y_q)

            # Update min_loss and best_alpha if the current loss is lower than min_loss
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_alpha = alpha.clone()

            if i % 100 == 0: 

                #print(f"Iteration {i}, Loss: {loss.item()}")
                print(f"Iteration {i}, Loss: {loss.item()}, Best loss: {min_loss}")


            loss.backward()
            optimizer.step()
            scheduler.step()

            

        
        return best_alpha   


        

def assign_ptf(model, model_size, cfg):
    print("Assign ptf...")
    if model_size == 'xxs':
        lr=1e-1
    else:
        lr=1e-2
    
    for name, module in model.named_modules():
 
                 
        if isinstance(module, BottleneckBlock):
            
            print(name + " Start!")
            alpha = quant_loss(module.conv1_1x1_olc_qact.olc_quantizer.observer, module.conv1_1x1.conv, lr=lr)
            module.conv1_1x1_olc.scale = alpha
            module.conv1_1x1.conv.weight.data = module.conv1_1x1.conv.weight.data * alpha

            alpha = quant_loss(module.conv3_1x1_olc_qact.olc_quantizer.observer, module.conv3_1x1.conv, lr=lr)
            module.conv3_1x1_olc.scale = alpha
            module.conv3_1x1.conv.weight.data = module.conv3_1x1.conv.weight.data * alpha
            
            
            module.qact1_1x1.quantizer.observer.min_val = torch.full([module.conv2_kxk.conv.in_channels],-0.27846455574035645, device = module.conv2_kxk.conv.weight.device)
            

                    
        
