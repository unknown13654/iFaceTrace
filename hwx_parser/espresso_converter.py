#!/usr/bin/env python3
'Legacy host helper; see README for precision and scope limits.'

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class EspressoParser:
    'Legacy host helper; see README for precision and scope limits.'

    def __init__(self, net_path, weights_path, shape_path=None):
        with open(net_path, 'r') as f:
            self.net = json.load(f)
        with open(weights_path, 'rb') as f:
            self.weights_raw = f.read()
        self.shapes = None
        if shape_path:
            with open(shape_path, 'r') as f:
                self.shapes = json.load(f)

        self.layers = self.net['layers']
        self._build_blob_map()

    def _build_blob_map(self):
        'Legacy host helper; see README for precision and scope limits.'
        import struct
        raw = self.weights_raw
        n_pairs = struct.unpack('<Q', raw[0:8])[0]
        header_size = 8 + n_pairs * 16

        self.blob_map = {}
        data_offset = header_size
        for i in range(n_pairs):
            base = 8 + i * 16
            blob_idx = struct.unpack('<Q', raw[base:base+8])[0]
            blob_size = struct.unpack('<Q', raw[base+8:base+16])[0]
            self.blob_map[blob_idx] = (data_offset, blob_size, None)
            data_offset += blob_size

    def read_blob(self, blob_index, as_f16=True):
        'Legacy host helper; see README for precision and scope limits.'
        if blob_index not in self.blob_map:
            raise KeyError(f"Blob index {blob_index} not found in blob_map ({len(self.blob_map)} entries)")
        offset, size, _ = self.blob_map[blob_index]
        dtype = np.float16 if as_f16 else np.float32
        arr = np.frombuffer(self.weights_raw[offset:offset + size], dtype=dtype).copy()
        # NaN → 0, Inf → ±65504 (fp16 max)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = 0.0
        inf_mask = np.isinf(arr)
        if inf_mask.any():
            arr[inf_mask] = np.sign(arr[inf_mask]) * np.finfo(np.float16).max
        return arr.astype(np.float32)

    def read_quantized_weights(self, layer):
        'Legacy host helper; see README for precision and scope limits.'
        C = layer.get('C', 0)


        w_dict = layer.get('weights', {})
        if 'W_S8' in w_dict:
            off_w, sz_w, _ = self.blob_map[w_dict['W_S8']]
            w_int8 = np.frombuffer(self.weights_raw[off_w:off_w + sz_w], dtype=np.int8)

            off_s, sz_s, _ = self.blob_map[w_dict['per_ch_qscale']]
            scales = np.frombuffer(self.weights_raw[off_s:off_s + sz_s], dtype=np.float32)

            off_b, sz_b, _ = self.blob_map[w_dict['per_ch_qbias']]
            biases = np.frombuffer(self.weights_raw[off_b:off_b + sz_b], dtype=np.float32)

            per_ch = w_int8.reshape(C, -1).astype(np.float32)
            weights = per_ch * scales[:C, None] + biases[:C, None]
            return weights.flatten()


        lut_idx = layer['quantization_lut_weights_blob']
        rng_idx = layer['quantization_ranges_blob']

        off_l, sz_l, _ = self.blob_map[lut_idx]
        off_r, sz_r, _ = self.blob_map[rng_idx]

        indices = np.frombuffer(self.weights_raw[off_l:off_l + sz_l], dtype=np.uint8)
        ranges = np.frombuffer(self.weights_raw[off_r:off_r + sz_r], dtype=np.float32)

        mins = ranges[:C]
        maxs = ranges[C:2*C]

        per_ch = indices.reshape(C, -1).astype(np.float32)
        weights = mins[:, None] + (per_ch / 255.0) * (maxs - mins)[:, None]
        return weights.flatten()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class OnnxGraphBuilder:
    'Legacy host helper; see README for precision and scope limits.'

    def __init__(self, parser, output_layers=None, fp16_sim=True):
        self.parser = parser
        self.output_layers = output_layers or ['flatten_output']
        self.fp16_sim = fp16_sim
        self.nodes = []
        self.initializers = []
        self.fc_outputs = set()

    def _add_fp16_cast(self, tensor_name):
        'Legacy host helper; see README for precision and scope limits.'
        if not self.fp16_sim:
            return tensor_name
        min_name = f'{tensor_name}_clip_min'
        max_name = f'{tensor_name}_clip_max'
        self.initializers.append(
            numpy_helper.from_array(np.array([-65504.0], dtype=np.float32), name=min_name))
        self.initializers.append(
            numpy_helper.from_array(np.array([65504.0], dtype=np.float32), name=max_name))
        clip_node = helper.make_node(
            'Clip', inputs=[tensor_name, min_name, max_name],
            outputs=[f'{tensor_name}_sim'],
            name=f'{tensor_name}_clip')
        self.nodes.append(clip_node)
        return f'{tensor_name}_sim'

    def build(self):
        'Legacy host helper; see README for precision and scope limits.'
        found_outputs = set()

        for layer in self.parser.layers:
            name = layer['name']
            lt = layer['type']
            bottom = layer.get('bottom', '')
            top = layer.get('top', name)

            if lt == 'convolution':
                self._add_conv(layer, name, bottom, top)
            elif lt == 'inner_product':
                self._add_inner_product(layer, name, bottom, top)
            elif lt == 'pool':
                self._add_pool(layer, name, bottom, top)
            elif lt == 'activation':
                self._add_activation(layer, name, bottom, top)
            elif lt == 'elementwise':
                self._add_elementwise(layer, name, bottom, top)
            elif lt == 'softmax':
                self._add_softmax(layer, name, bottom, top)
            elif lt == 'concat':
                self._add_concat(layer, name, bottom, top)
            elif lt == 'deconvolution':
                self._add_deconv(layer, name, bottom, top)
            elif lt == 'batchnorm':
                self._add_batchnorm(layer, name, bottom, top)
            elif lt == 'load_constant':
                self._add_load_constant(layer, name, bottom, top)
            elif lt == 'upsample':
                self._add_upsample(layer, name, bottom, top)
            elif lt == 'transpose':
                self._add_transpose(layer, name, bottom, top)
            elif lt == 'reshape':
                self._add_reshape(layer, name, bottom, top)
            elif lt == 'slice':
                self._add_slice(layer, name, bottom, top)
            elif lt == 'reduce':
                self._add_reduce(layer, name, bottom, top)
            elif lt == 'non_maximum_suppression':
                pass

            if top in self.output_layers:
                found_outputs.add(top)
            if found_outputs == set(self.output_layers):
                break

    def _add_initializer(self, name, data, shape=None):
        'Legacy host helper; see README for precision and scope limits.'
        if shape is not None:
            data = data.reshape(shape)
        tensor = numpy_helper.from_array(data, name=name)
        self.initializers.append(tensor)

    def _add_conv(self, layer, name, bottom, top):
        K = layer['K']
        C = layer['C']
        Nx = layer.get('Nx', 1)
        Ny = layer.get('Ny', 1)
        ng = layer.get('n_groups', 1)
        sx = layer.get('stride_x', 1)
        sy = layer.get('stride_y', 1)
        pt = layer.get('pad_t', 0)
        pb = layer.get('pad_b', 0)
        pl = layer.get('pad_l', 0)
        pr = layer.get('pad_r', 0)
        has_biases = bool(layer.get('has_biases', 0))
        fused_relu = bool(layer.get('fused_relu', 0))



        if 'blob_weights_f16' in layer:
            w_data = self.parser.read_blob(layer['blob_weights_f16'], as_f16=True)
        elif 'blob_weights' in layer:
            w_data = self.parser.read_blob(layer['blob_weights'], as_f16=False)
        elif 'quantization_lut_weights_blob' in layer or 'W_S8' in layer.get('weights', {}):
            w_data = self.parser.read_quantized_weights(layer)
        else:
            raise KeyError(f"No weight blob found in layer {name}")
        w_name = f'{name}_W'
        self._add_initializer(w_name, w_data, shape=(C, K // ng, Ny, Nx))

        inputs = [bottom, w_name]
        if has_biases:
            b_key = 'blob_biases_f16' if 'blob_biases_f16' in layer else 'blob_biases'
            if layer.get(b_key) is not None:
                b_is_f16 = 'blob_biases_f16' in layer
                b_data = self.parser.read_blob(layer[b_key], as_f16=b_is_f16)
            b_name = f'{name}_B'
            self._add_initializer(b_name, b_data, shape=(C,))
            inputs.append(b_name)


        raw_out = f'{top}_raw'

        # pad_mode: 0=explicit, 1=SAME, 2=VALID
        pad_mode = layer.get('pad_mode', 0)
        conv_attrs = dict(
            kernel_shape=[Ny, Nx],
            strides=[sy, sx],
            group=ng,
        )
        if pad_mode == 1:
            conv_attrs['auto_pad'] = 'SAME_UPPER'
        elif pad_mode == 2:
            conv_attrs['auto_pad'] = 'VALID'
        else:
            conv_attrs['pads'] = [pt, pl, pb, pr]

        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[raw_out],
            name=f'{name}_conv',
            **conv_attrs,
        )
        self.nodes.append(conv_node)


        sim_out = self._add_fp16_cast(raw_out)

        if fused_relu:
            relu_node = helper.make_node(
                'Relu',
                inputs=[sim_out],
                outputs=[top],
                name=f'{name}_relu',
            )
            self.nodes.append(relu_node)
        else:

            id_node = helper.make_node(
                'Identity', inputs=[sim_out], outputs=[top],
                name=f'{name}_id')
            self.nodes.append(id_node)

    def _add_inner_product(self, layer, name, bottom, top):
        nB = layer.get('nB', 0)
        nC = layer.get('nC', 0)
        has_biases = bool(layer.get('has_biases', 0))
        has_relu = bool(layer.get('has_relu', 0))


        actual_bottom = bottom
        if bottom not in self.fc_outputs:
            flat_name = f'{name}_flatten_out'
            flat_node = helper.make_node(
                'Flatten',
                inputs=[bottom],
                outputs=[flat_name],
                name=f'{name}_flatten',
                axis=1,
            )
            self.nodes.append(flat_node)
            actual_bottom = flat_name


        if 'blob_weights_f16' in layer:
            w_data = self.parser.read_blob(layer['blob_weights_f16'], as_f16=True)
        else:
            w_data = self.parser.read_blob(layer['blob_weights'], as_f16=False)
        w_name = f'{name}_W'
        self._add_initializer(w_name, w_data, shape=(nC, nB))

        gemm_out = f'{top}_pre_relu' if has_relu else top
        inputs = [actual_bottom, w_name]

        if has_biases:
            b_key = 'blob_biases_f16' if 'blob_biases_f16' in layer else 'blob_biases'
            if layer.get(b_key) is not None:
                b_is_f16 = 'blob_biases_f16' in layer
                b_data = self.parser.read_blob(layer[b_key], as_f16=b_is_f16)
            b_name = f'{name}_B'
            self._add_initializer(b_name, b_data, shape=(nC,))
            inputs.append(b_name)

        raw_out = f'{gemm_out}_raw' if has_relu else f'{top}_raw'

        gemm_node = helper.make_node(
            'Gemm',
            inputs=inputs,
            outputs=[raw_out],
            name=f'{name}_gemm',
            alpha=1.0,
            beta=1.0,
            transB=1,
        )
        self.nodes.append(gemm_node)


        sim_out = self._add_fp16_cast(raw_out)

        if has_relu:
            relu_node = helper.make_node(
                'Relu',
                inputs=[sim_out],
                outputs=[top],
                name=f'{name}_relu',
            )
            self.nodes.append(relu_node)
        else:
            id_node = helper.make_node(
                'Identity', inputs=[sim_out], outputs=[top],
                name=f'{name}_id')
            self.nodes.append(id_node)

        self.fc_outputs.add(top)

    def _add_pool(self, layer, name, bottom, top):
        is_global = bool(layer.get('is_global', 0))
        avg_or_max = layer.get('avg_or_max', 0)

        if is_global:
            op = 'GlobalAveragePool' if avg_or_max == 0 else 'GlobalMaxPool'
            node = helper.make_node(
                op,
                inputs=[bottom],
                outputs=[top],
                name=name,
            )
        else:
            sx = layer.get('stride_x', 1)
            sy = layer.get('stride_y', 1)
            kx = layer.get('size_x', 1)
            ky = layer.get('size_y', 1)
            op = 'AveragePool' if avg_or_max == 0 else 'MaxPool'
            # pad_mode: 0=explicit, 1=SAME, 2=VALID
            pad_mode = layer.get('pad_mode', 0)
            pool_attrs = dict(
                kernel_shape=[ky, kx],
                strides=[sy, sx],
            )
            if pad_mode == 1:
                pool_attrs['auto_pad'] = 'SAME_UPPER'
            elif pad_mode == 2:
                pool_attrs['auto_pad'] = 'VALID'
            else:
                pt = layer.get('pad_t', 0)
                pb = layer.get('pad_b', 0)
                pl = layer.get('pad_l', 0)
                pr = layer.get('pad_r', 0)
                pool_attrs['pads'] = [pt, pl, pb, pr]
            node = helper.make_node(
                op,
                inputs=[bottom],
                outputs=[top],
                name=name,
                **pool_attrs,
            )
        self.nodes.append(node)

    def _add_activation(self, layer, name, bottom, top):
        mode = layer.get('mode', 0)
        if mode == 0:
            op = 'Relu'
        elif mode == 1:
            op = 'Tanh'
        elif mode == 3:
            op = 'Sigmoid'
        else:
            op = 'Identity'

        node = helper.make_node(op, inputs=[bottom], outputs=[top], name=name)
        self.nodes.append(node)

    def _add_elementwise(self, layer, name, bottom, top):
        """Espresso elementwise: output = alpha * op(inputs) + beta"""
        bottoms = [b.strip() for b in bottom.split(',')]
        operation = layer.get('operation', 0)
        alpha = layer.get('alpha', 1)
        beta = layer.get('beta', 0)

        if operation == 0:
            op = 'Add'
        elif operation == 1:
            op = 'Mul'
        elif operation == 2:
            op = 'Max'
        else:
            op = 'Add'


        if len(bottoms) == 1 and (alpha != 1 or beta != 0):
            current = bottoms[0]
            if alpha != 1:
                a_name = f'{name}_alpha'
                self._add_initializer(a_name, np.array([alpha], dtype=np.float32))
                mul_out = f'{top}_amul' if beta != 0 else top
                self.nodes.append(helper.make_node(
                    'Mul', inputs=[current, a_name], outputs=[mul_out], name=f'{name}_amul'))
                current = mul_out
            if beta != 0:
                b_name = f'{name}_beta'
                self._add_initializer(b_name, np.array([beta], dtype=np.float32))
                self.nodes.append(helper.make_node(
                    'Add', inputs=[current, b_name], outputs=[top], name=f'{name}_badd'))
            return


        need_post = (alpha != 1 or beta != 0)
        op_out = f'{top}_op' if need_post else top
        node = helper.make_node(op, inputs=bottoms, outputs=[op_out], name=f'{name}_op' if need_post else name)
        self.nodes.append(node)

        if need_post:
            current = op_out
            if alpha != 1:
                a_name = f'{name}_alpha'
                self._add_initializer(a_name, np.array([alpha], dtype=np.float32))
                mul_out = f'{top}_amul' if beta != 0 else top
                self.nodes.append(helper.make_node(
                    'Mul', inputs=[current, a_name], outputs=[mul_out], name=f'{name}_amul'))
                current = mul_out
            if beta != 0:
                b_name = f'{name}_beta'
                self._add_initializer(b_name, np.array([beta], dtype=np.float32))
                self.nodes.append(helper.make_node(
                    'Add', inputs=[current, b_name], outputs=[top], name=f'{name}_badd'))
            return

    def _add_softmax(self, layer, name, bottom, top):
        node = helper.make_node(
            'Softmax',
            inputs=[bottom],
            outputs=[top],
            name=name,
            axis=1,
        )
        self.nodes.append(node)

    def _add_batchnorm(self, layer, name, bottom, top):
        'Legacy host helper; see README for precision and scope limits.'
        C = layer['C']
        blob_idx = layer['blob_batchnorm_params']
        offset, size, _ = self.parser.blob_map[blob_idx]
        params = np.frombuffer(
            self.parser.weights_raw[offset:offset + size], dtype=np.float32).copy()

        n_per_ch = len(params) // C
        if n_per_ch == 4:
            # Interleaved: [scale, bias, mean, var] per channel
            interleaved = params.reshape(C, 4)
            scale = interleaved[:, 0].copy()
            bias = interleaved[:, 1].copy()
        elif n_per_ch == 2:
            scale = params[:C].copy()
            bias = params[C:].copy()
        else:
            raise ValueError(f"Unsupported batchnorm params: {len(params)} values for C={C}")

        scale_name = f'{name}_scale'
        bias_name = f'{name}_bias'
        self._add_initializer(scale_name, scale, shape=(C,))
        self._add_initializer(bias_name, bias, shape=(C,))
        mul_out = f'{top}_mul'
        self.nodes.append(helper.make_node(
            'Mul', inputs=[bottom, scale_name], outputs=[mul_out], name=f'{name}_mul'))
        self.nodes.append(helper.make_node(
            'Add', inputs=[mul_out, bias_name], outputs=[top], name=f'{name}_add'))

    def _add_concat(self, layer, name, bottom, top):
        'Legacy host helper; see README for precision and scope limits.'
        bottoms = [b.strip() for b in bottom.split(',')]
        axis = 3 if layer.get('nd_mode', False) else 1
        node = helper.make_node(
            'Concat', inputs=bottoms, outputs=[top], name=name, axis=axis)
        self.nodes.append(node)

    def _add_deconv(self, layer, name, bottom, top):
        """Espresso deconvolution (ConvTranspose)."""
        K = layer['K']
        C = layer['C']
        Nx = layer.get('Nx', 1)
        Ny = layer.get('Ny', 1)
        ng = layer.get('n_groups', 1)
        sx = layer.get('stride_x', 1)
        sy = layer.get('stride_y', 1)
        has_biases = bool(layer.get('has_biases', 0))


        if 'blob_weights_f16' in layer:
            w_data = self.parser.read_blob(layer['blob_weights_f16'], as_f16=True)
        elif 'blob_weights' in layer:
            w_data = self.parser.read_blob(layer['blob_weights'], as_f16=False)
        elif 'quantization_lut_weights_blob' in layer:
            w_data = self.parser.read_quantized_weights(layer)
        else:
            raise KeyError(f"No weight blob found in deconv layer {name}")
        w_name = f'{name}_W'
        self._add_initializer(w_name, w_data, shape=(K, C // ng, Ny, Nx))

        inputs = [bottom, w_name]
        if has_biases:
            b_key = 'blob_biases_f16' if 'blob_biases_f16' in layer else 'blob_biases'
            if layer.get(b_key) is not None:
                b_is_f16 = 'blob_biases_f16' in layer
                b_data = self.parser.read_blob(layer[b_key], as_f16=b_is_f16)
                b_name = f'{name}_B'
                self._add_initializer(b_name, b_data, shape=(C,))
                inputs.append(b_name)

        pad_mode = layer.get('pad_mode', 0)
        deconv_attrs = dict(
            kernel_shape=[Ny, Nx],
            strides=[sy, sx],
            group=ng,
        )
        if pad_mode == 1:
            deconv_attrs['auto_pad'] = 'SAME_UPPER'
        elif pad_mode == 2:
            deconv_attrs['auto_pad'] = 'VALID'
        else:
            pt = layer.get('pad_t', 0)
            pb = layer.get('pad_b', 0)
            pl = layer.get('pad_l', 0)
            pr = layer.get('pad_r', 0)
            deconv_attrs['pads'] = [pt, pl, pb, pr]


        out_h = layer.get('deconv_out_height')
        out_w = layer.get('deconv_out_width')
        if out_h and out_w:
            deconv_attrs['output_shape'] = [out_h, out_w]

        node = helper.make_node(
            'ConvTranspose',
            inputs=inputs,
            outputs=[top],
            name=f'{name}_deconv',
            **deconv_attrs,
        )
        self.nodes.append(node)


    def _add_load_constant(self, layer, name, bottom, top):
        'Legacy host helper; see README for precision and scope limits.'
        blob_idx = layer['constant_blob']

        offset, size, _ = self.parser.blob_map[blob_idx]
        n = layer.get('n', 1)
        k = layer.get('k', 1)
        h = layer.get('h', 1)
        w = layer.get('w', 1)
        expected_count = max(n * k * h * w, 1)
        is_f16 = (size == expected_count * 2)
        data = self.parser.read_blob(blob_idx, as_f16=is_f16)
        shape = (n, k, h, w)
        self._add_initializer(top, data, shape=shape)

    def _add_upsample(self, layer, name, bottom, top):
        """2x nearest-neighbor upsampling → ONNX Resize."""
        sx = layer.get('scaling_factor_x', 2)
        sy = layer.get('scaling_factor_y', 2)
        # Resize requires empty ROI and scales tensors
        roi_name = f'{name}_roi'
        scales_name = f'{name}_scales'
        self._add_initializer(roi_name, np.array([], dtype=np.float32))
        self._add_initializer(scales_name, np.array([1.0, 1.0, float(sy), float(sx)], dtype=np.float32))
        node = helper.make_node(
            'Resize',
            inputs=[bottom, roi_name, scales_name],
            outputs=[top],
            name=name,
            mode='nearest',
            coordinate_transformation_mode='asymmetric',
        )
        self.nodes.append(node)

    def _add_transpose(self, layer, name, bottom, top):
        """Espresso transpose → ONNX Transpose.

        Verified: axis_h=0,k=1,w=2,n=3 → NCHW→NHWC → perm=[0,2,3,1]
        (shape file: input (1,12,56,76) → output (1,56,76,12), k=56,h=76,w=12)
        """
        ah = layer.get('axis_h', 0)
        ak = layer.get('axis_k', 1)
        aw = layer.get('axis_w', 2)
        an = layer.get('axis_n', 3)

        if (ah, ak, aw, an) == (0, 1, 2, 3):
            perm = [0, 2, 3, 1]  # NCHW → NHWC
        else:
            raise ValueError(f"Unsupported transpose axes: h={ah} k={ak} w={aw} n={an}")

        node = helper.make_node(
            'Transpose', inputs=[bottom], outputs=[top], name=name, perm=perm)
        self.nodes.append(node)

    def _add_reshape(self, layer, name, bottom, top):
        """Espresso reshape → ONNX Reshape with shape tensor."""
        dst_n = layer.get('dst_n', 0)
        dst_k = layer.get('dst_k', 0)
        dst_h = layer.get('dst_h', 0)
        dst_w = layer.get('dst_w', 0)
        dst_nd_rank = layer.get('dst_nd_rank', 4)

        # ONNX shape: [N, C, H, W]



        n_val = 1 if dst_nd_rank <= 3 else dst_n
        k_val = 1 if dst_nd_rank <= 2 else dst_k
        h_val = dst_h
        w_val = dst_w

        shape = np.array([n_val, k_val, h_val, w_val], dtype=np.int64)
        shape_name = f'{name}_shape'
        tensor = numpy_helper.from_array(shape, name=shape_name)
        self.initializers.append(tensor)
        node = helper.make_node(
            'Reshape', inputs=[bottom, shape_name], outputs=[top], name=name)
        self.nodes.append(node)

    def _add_slice(self, layer, name, bottom, top):
        """Espresso slice → ONNX Slice."""
        axis = layer.get('axis', 0)
        start = layer.get('start', 0)
        end = layer.get('end', 0)
        # Espresso axis → ONNX axis: 0(W)→3, 2(C)→1
        onnx_axis = 3 - axis

        for suffix, arr in [('starts', [start]), ('ends', [end]),
                            ('axes', [onnx_axis]), ('steps', [1])]:
            t_name = f'{name}_{suffix}'
            tensor = numpy_helper.from_array(
                np.array(arr, dtype=np.int64), name=t_name)
            self.initializers.append(tensor)

        node = helper.make_node(
            'Slice',
            inputs=[bottom, f'{name}_starts', f'{name}_ends', f'{name}_axes', f'{name}_steps'],
            outputs=[top], name=name)
        self.nodes.append(node)

    def _add_reduce(self, layer, name, bottom, top):
        """Espresso reduce → ONNX ReduceMax (mode=9) or ArgMax (mode=11)."""
        mode = layer.get('mode', 0)
        # axis_mode=1 → reduce along last meaningful dim → ONNX axis 3
        onnx_axis = 3

        if mode == 9:  # ReduceMax
            node = helper.make_node(
                'ReduceMax', inputs=[bottom], outputs=[top], name=name,
                axes=[onnx_axis], keepdims=1)
            self.nodes.append(node)
        elif mode == 11:  # ArgMax → int64 output, cast to float32
            argmax_out = f'{top}_i64'
            node = helper.make_node(
                'ArgMax', inputs=[bottom], outputs=[argmax_out], name=name,
                axis=onnx_axis, keepdims=1)
            self.nodes.append(node)
            cast_node = helper.make_node(
                'Cast', inputs=[argmax_out], outputs=[top],
                name=f'{name}_cast', to=TensorProto.FLOAT)
            self.nodes.append(cast_node)
        else:
            raise ValueError(f"Unsupported reduce mode: {mode}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def convert_espresso_to_onnx(net_path, weights_path, shape_path, onnx_path,
                              output_layers=None, fp16_sim=True):
    'Legacy host helper; see README for precision and scope limits.'
    output_layers = output_layers or ['flatten_output']

    print('Legacy host helper; see README for precision and scope limits.')
    parser = EspressoParser(net_path, weights_path, shape_path)
    print(f'Legacy host report: {len(parser.layers)}')
    print(f'Legacy host report: {len(parser.blob_map)}')

    print(f"Legacy host report: {('ON' if fp16_sim else 'OFF')})")
    builder = OnnxGraphBuilder(parser, output_layers=output_layers, fp16_sim=fp16_sim)
    builder.build()
    print(f'Legacy host report: {len(builder.nodes)}')
    print(f'Legacy host report: {len(builder.initializers)}')


    first_layer = parser.layers[0]
    input_name = first_layer.get('bottom', 'data')
    in_channels = first_layer.get('K', 3)
    in_h, in_w = 128, 128
    if parser.shapes:

        layer_shapes = parser.shapes
        if isinstance(layer_shapes, dict) and 'layer_shapes' in layer_shapes:
            layer_shapes = layer_shapes['layer_shapes']
        if isinstance(layer_shapes, dict) and input_name in layer_shapes:
            s = layer_shapes[input_name]
            in_h = s.get('h', 128)
            in_w = s.get('w', 128)
            in_channels = s.get('k', in_channels)
    input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, in_channels, in_h, in_w])

    output_infos = []
    for name in output_layers:
        output_infos.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        )

    graph = helper.make_graph(
        builder.nodes,
        Path(net_path).stem.replace('.espresso', '').replace('.', '_'),
        [input_info],
        output_infos,
        initializer=builder.initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8

    print('Legacy host helper; see README for precision and scope limits.')
    try:
        onnx.checker.check_model(model)
        print('Legacy host helper; see README for precision and scope limits.')
    except onnx.checker.ValidationError as e:
        print(f'Legacy host report: {e}')

    onnx.save(model, str(onnx_path))
    file_size = Path(onnx_path).stat().st_size
    print(f'Legacy host report: {onnx_path} ({file_size:,} bytes)')


    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        actual_input_name = session.get_inputs()[0].name
        results = session.run(output_layers, {actual_input_name: dummy})
        print('Legacy host helper; see README for precision and scope limits.')
        for name, result in zip(output_layers, results):
            print(f"        {name}: shape={result.shape}, "
                  f"range=[{result.min():.4f}, {result.max():.4f}]")
    except Exception as e:
        print(f'Legacy host report: {e}')

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Espresso → ONNX converter")
    parser.add_argument('--net', required=True, help='Path to .espresso.net')
    parser.add_argument('--weights', required=True, help='Path to .espresso.weights_nonane')
    parser.add_argument('--shape', default=None, help='Path to .espresso.shape')
    parser.add_argument('--output', '-o', required=True, help='Output ONNX path')
    parser.add_argument('--output-layers', nargs='+', default=['flatten_output'],
                        help='Output layer names (default: flatten_output)')
    parser.add_argument('--no-fp16-sim', action='store_true',
                        help='Disable fp16 saturation simulation (Clip nodes)')
    args = parser.parse_args()

    convert_espresso_to_onnx(
        args.net, args.weights, args.shape, args.output,
        output_layers=args.output_layers,
        fp16_sim=not args.no_fp16_sim,
    )


if __name__ == '__main__':
    main()
