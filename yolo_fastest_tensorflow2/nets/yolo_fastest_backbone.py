from functools import wraps
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Lambda, Layer, LeakyReLU, BatchNormalization, SeparableConv2D
from tensorflow.keras.regularizers import l2
from utils.utils import compose
import tensorflow as tf

#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # 多了一个正则化的项
    # 正则化系数5e-4
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DepthwiseConv2D(*args, **kwargs):
    # 多了一个正则化的项
    # 正则化系数5e-4
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def DepthwiseConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DepthwiseConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))
#---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
def resblock_body(x, num_filters, num_filters_1):
    route = x
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    #x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(num_filters_1, (3, 3))(x)
    #x = DarknetConv2D_BN_Leaky(num_filters_1, (1, 1))(x)
    #x = route + x
    x = Concatenate()([route, x])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    # 进行长和宽的压缩，下一步步长为2
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    # 416,416,3 -> 208,208,32
    x = DepthwiseConv2D_BN_Leaky(8, (3, 3), strides=(2, 2))(x)
    #x = DarknetConv2D_BN_Leaky(8, (1, 1), strides=(1, 1))(x)
    #x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(4, (3, 3), strides=(1, 1))(x)
    #x = DarknetConv2D_BN_Leaky(4, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 8, 4)
    x = DarknetConv2D_BN_Leaky(24, (1, 1), strides=(1, 1))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(8, (3, 3), strides=(2, 2))(x)
    #x = DarknetConv2D_BN_Leaky(8, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 32, 8)
    x = resblock_body(x, 32, 8)
    x = DarknetConv2D_BN_Leaky(32, (1, 1), strides=(1, 1))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(8, (3, 3), strides=(2, 2))(x)
    #x = DarknetConv2D_BN_Leaky(8, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 48, 8)
    x = resblock_body(x, 48, 8)
    x = DarknetConv2D_BN_Leaky(48, (1, 1), strides=(1, 1))(x)
    x = DepthwiseConv2D_BN_Leaky(16, (3, 3), strides=(1, 1))(x)
    #x = DarknetConv2D_BN_Leaky(16, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 96, 16)
    x = resblock_body(x, 96, 16)
    x = resblock_body(x, 96, 16)
    x = resblock_body(x, 96, 16)
    x = DarknetConv2D_BN_Leaky(96, (1, 1), strides=(1, 1))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(24, (3, 3), strides=(2, 2))(x)
    #x = DarknetConv2D_BN_Leaky(24, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 136, 24)
    x = resblock_body(x, 136, 24)
    x = resblock_body(x, 136, 24)
    x = resblock_body(x, 136, 24)
    x = DarknetConv2D_BN_Leaky(136, (1, 1), strides=(1, 1))(x)
    feat1 = x
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DepthwiseConv2D_BN_Leaky(48, (3, 3), strides=(2, 2))(x)
    #x = DarknetConv2D_BN_Leaky(48, (1, 1), strides=(1, 1))(x)
    x = resblock_body(x, 224, 48)
    x = resblock_body(x, 224, 48)
    x = resblock_body(x, 224, 48)
    x = resblock_body(x, 224, 48)
    x = resblock_body(x, 224, 48)
    x = DarknetConv2D_BN_Leaky(96, (1, 1), strides=(1, 1))(x)
    feat2 = x
    return feat1, feat2
