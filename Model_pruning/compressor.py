# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import numpy as np
from . import default_layers
from kerassurgeon import Surgeon
tf.config.experimental_run_functions_eagerly(True)

_logger = logging.getLogger(__name__)


class LayerInfo:
    def __init__(self, keras_layer):
        self.keras_layer = keras_layer
        self.name = keras_layer.name
        self.type = default_layers.get_op_type(type(keras_layer))
        self.weight_index = default_layers.get_weight_index(self.type)
        if self.weight_index is not None:
            self.weight = keras_layer.weights[self.weight_index]
        self._call = None

class Compressor:
    """
    Abstract base TensorFlow compressor
    """

    def __init__(self, model, config_list):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : keras model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        """
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = []
        self.modules_to_delete = []

    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.

        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is not None:
            self.modules_to_compress = []
            for keras_layer in self.bound_model.layers:
                layer = LayerInfo(keras_layer)
                config = self.select_config(layer)
                if (config is not None) and (layer.name == 'conv2d_50'):    #Choose which layer to prune
                    self.modules_to_compress.append((layer, config))
                    self.modules_to_delete.append((keras_layer, config))
        return self.modules_to_compress, self.modules_to_delete

    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        modules_to_compress, ignore_ = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            self._instrument_layer(layer, config)
        return self.bound_model

    def compress_model(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        ignore_, modules_to_compress = self.detect_modules_to_compress()
        for layer,config in modules_to_compress:
            layer_1 = LayerInfo(layer)
            self.bound_model = self.Prun_channel(layer_1, layer, config)
            #a_list = self.Prun_channel(layer_1, layer, config)
        return self.bound_model

    def compress_model_1(self, channels_p):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        ignore_, modules_to_compress = self.detect_modules_to_compress()
        for layer,config in modules_to_compress:
            layer_1 = LayerInfo(layer)
            #self.bound_model = self.Prun_channel(layer_1, layer, config)
            self.bound_model = self.Prun_channel_1(layer_1, layer, config, channels_p)
        return self.bound_model

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed layers.

        Returns
        -------
        self.modules_to_compress : list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer: LayerInfo
            one layer

        Returns
        -------
        ret : config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        if layer.type is None:
            return None
        for config in self.config_list:
            config = config.copy()
            config['op_types'] = self._expand_config_op_types(config)
            if layer.type not in config['op_types']:
                continue
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret

    def update_epoch(self, epoch):
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """

    def step(self):
        """
        If user want to update mask every step, user can override this method
        """


    def _instrument_layer(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def Prun_channel(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def Prun_channel_1(self, layer, config, channels_p):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def _expand_config_op_types(self, config):
        if config is None:
            return []
        op_types = []

        for op_type in config.get('op_types', []):
            if op_type == 'default':
                op_types.extend(default_layers.default_layers)
            else:
                op_types.append(op_type)
        return op_types


class Pruner(Compressor):
    """
    Abstract base TensorFlow pruner
    """

    def calc_mask(self, layer, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        layer._call = layer.keras_layer.call

        def new_call(*inputs):
            weights = [x.numpy() for x in layer.keras_layer.weights]
            mask = self.calc_mask(layer, config)
            weights[layer.weight_index] = weights[layer.weight_index] * mask
            layer.keras_layer.set_weights(weights)
            ret = layer._call(*inputs)
            return ret

        layer.keras_layer.call = new_call

    def Prun_channel(self, layer, layer_1, config):
        weight = layer.weight
        op_type = layer.type
        op_name = layer.name
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv1D', 'Conv2D']
        assert op_type in config['op_types']

        # op_name = layer.name
        # assert 0 <= config.get('sparsity') < 1
        # assert op_type in ['Conv1D', 'Conv2D']
        # assert op_type in config['op_types']

        if layer.name in self.epoch_pruned_layers:
            assert layer.name in self.mask_dict
            return self.mask_dict.get(layer.name)

        try:
            w = tf.stop_gradient(tf.transpose(tf.reshape(weight, (-1, weight.shape[-1])), [1, 0]))
            masks = np.ones(w.shape)
            num_filters = w.shape[0]
            num_prune = int(num_filters * config.get('sparsity'))
            if num_filters < 2 or num_prune < 1:
                return masks
            min_gm_idx = self._get_min_gm_kernel_idx_m(w, num_prune)

            surgeon = Surgeon(self.bound_model, copy=False)
            channels = min_gm_idx
            surgeon.add_job('delete_channels', layer_1, channels=channels)

            #for idx in min_gm_idx:
            #    masks[idx] = 0.
        finally:
            masks = tf.reshape(tf.transpose(masks, [1, 0]), weight.shape)
            masks = tf.Variable(masks)
            self.mask_dict.update({op_name: masks})
            self.epoch_pruned_layers.add(layer.name)

        return surgeon.operate()
        #return min_gm_idx


    def Prun_channel_1(self, layer, layer_1, config, channels_p):
        weight = layer.weight
        op_type = layer.type
        op_name = layer.name
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv1D', 'Conv2D']
        assert op_type in config['op_types']

        # op_name = layer.name
        # assert 0 <= config.get('sparsity') < 1
        # assert op_type in ['Conv1D', 'Conv2D']
        # assert op_type in config['op_types']

        #if layer.name in self.epoch_pruned_layers:
        #    assert layer.name in self.mask_dict
        #    return self.mask_dict.get(layer.name)

        try:
            w = tf.stop_gradient(tf.transpose(tf.reshape(weight, (-1, weight.shape[-1])), [1, 0]))
            masks = np.ones(w.shape)
            num_filters = w.shape[0]
            num_prune = int(num_filters * config.get('sparsity'))
            if num_filters < 2 or num_prune < 1:
                return masks
            #min_gm_idx = self._get_min_gm_kernel_idx_m(w, num_prune)

            surgeon = Surgeon(self.bound_model, copy=False)
            channels = channels_p
            surgeon.add_job('delete_channels', layer_1, channels=channels)

            #for idx in min_gm_idx:
            #    masks[idx] = 0.
        finally:
            masks = tf.reshape(tf.transpose(masks, [1, 0]), weight.shape)
            masks = tf.Variable(masks)
            self.mask_dict.update({op_name: masks})
            self.epoch_pruned_layers.add(layer.name)

        return surgeon.operate()
        #return min_gm_idx


    def _get_min_gm_kernel_idx_m(self, weight, n):
        
        dist_list = []
        sum_max = 0;
        for out_i in range(weight.shape[0]):
            dist_sum = self._get_distance_sum_m(weight, out_i)
            dist_list.append((dist_sum, out_i))
            #dist_list.append(dist_sum)

        a=0
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:n]

        #min_gm_kernels = dist_list[:n]
        #size_a = tf.size(dist_list)
        #min_gm_kernels = tf.sort(dist_list)
        return [x[1] for x in min_gm_kernels]
        #return dist_list

    def _get_distance_sum_m(self, weight, out_idx):
        anchor_w = tf.tile(tf.expand_dims(weight[out_idx], 0), [weight.shape[0], 1])
        x = weight - anchor_w
        x = tf.math.reduce_sum((x*x), -1)
        x = tf.math.sqrt(x)
        return tf.math.reduce_sum(x)


class Quantizer(Compressor):
    """
    Abstract base TensorFlow quantizer
    """

    def quantize_weight(self, weight, config, op, op_type, op_name):
        raise NotImplementedError("Quantizer must overload quantize_weight()")
