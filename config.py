from models import BIT_TYPE_DICT


class Config:

    def __init__(self, ptf=True, lis=True, olc=True, quant_method='minmax', int_sm=False, shift_swish=False):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_S16 = BIT_TYPE_DICT['int16']

        self.BIT_TYPE_U2 = BIT_TYPE_DICT['uint2']
        self.BIT_TYPE_U4 = BIT_TYPE_DICT['uint4']
        self.BIT_TYPE_U2 = BIT_TYPE_DICT['uint2']
        self.BIT_TYPE_U6 = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_U16 = BIT_TYPE_DICT['uint16']
        self.BIT_TYPE_U32 = BIT_TYPE_DICT['uint32']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        # DOING
        self.QUANTIZER_A_OLC = 'uniform'
        #self.CALIBRATION_MODE_OLC = 'channel_wise'

        self.shift_swish = False;

        if(shift_swish):
            self.shift_swish = True
        else:
            self.shift_swish = False
            
        if olc:
            self.OLC = True
            self.OBSERVER_A_OLC = 'olc'
            self.CALIBRATION_MODE_A_OLC = self.CALIBRATION_MODE_A
        else:
            self.OLC = False
            self.OBSERVER_A_OLC = self.OBSERVER_A
            self.CALIBRATION_MODE_A_OLC = self.CALIBRATION_MODE_A
            
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        # elif int_sm:
        #     self.INT_SOFTMAX = True
        #     self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
        #     self.OBSERVER_S = 'minmax'
        #     self.QUANTIZER_S = self.QUANTIZER_A
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
            
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
            