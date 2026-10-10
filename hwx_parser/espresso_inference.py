#!/usr/bin/env python3
'Legacy host helper; see README for precision and scope limits.'

import json
import sys
from pathlib import Path

import numpy as np


class EspressoParser:
    'Legacy host helper; see README for precision and scope limits.'

    def __init__(self, net_path, weights_path):
        with open(net_path, 'r') as f:
            self.net = json.load(f)
        with open(weights_path, 'rb') as f:
            self.weights_raw = f.read()
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

    def read_blob_f16(self, blob_index, as_f16=True):
        'Legacy host helper; see README for precision and scope limits.'
        offset, size, dtype_str = self.blob_map[blob_index]
        dtype = np.float16 if as_f16 else np.float32
        arr = np.frombuffer(self.weights_raw[offset:offset + size], dtype=dtype).copy()
        arr[np.isnan(arr)] = 0
        inf_mask = np.isinf(arr)
        if inf_mask.any():
            arr[inf_mask] = np.sign(arr[inf_mask]) * np.finfo(np.float16).max
        return arr.astype(np.float16)


def _conv2d_fp16(x, weight, bias, stride, padding, groups):
    'Legacy host helper; see README for precision and scope limits.'
    # x: (1, C_in, H, W) fp16
    # weight: (C_out, C_in/groups, kH, kW) fp16
    _, C_in, H, W = x.shape
    C_out, C_grp, kH, kW = weight.shape


    pt, pl, pb, pr = padding
    if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)),
                    mode='constant', constant_values=0)

    _, _, H_pad, W_pad = x.shape
    sy, sx = stride
    H_out = (H_pad - kH) // sy + 1
    W_out = (W_pad - kW) // sx + 1


    x_f32 = x.astype(np.float32)
    w_f32 = weight.astype(np.float32)

    out = np.zeros((1, C_out, H_out, W_out), dtype=np.float32)

    if groups == 1:

        # im2col
        cols = np.zeros((C_in * kH * kW, H_out * W_out), dtype=np.float32)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x_f32[0, :, i*sy:i*sy+kH, j*sx:j*sx+kW]
                cols[:, idx] = patch.reshape(-1)
                idx += 1
        # GEMM: (C_out, C_in*kH*kW) × (C_in*kH*kW, H_out*W_out)
        w_mat = w_f32.reshape(C_out, -1)
        result = w_mat @ cols  # (C_out, H_out*W_out)
        out[0] = result.reshape(C_out, H_out, W_out)
    elif groups == C_in and groups == C_out:

        for c in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x_f32[0, c, i*sy:i*sy+kH, j*sx:j*sx+kW]
                    out[0, c, i, j] = np.sum(patch * w_f32[c, 0])
    else:

        c_per_group_in = C_in // groups
        c_per_group_out = C_out // groups
        for g in range(groups):
            in_start = g * c_per_group_in
            in_end = in_start + c_per_group_in
            out_start = g * c_per_group_out
            out_end = out_start + c_per_group_out
            x_g = x_f32[:, in_start:in_end]
            w_g = w_f32[out_start:out_end]
            for oc in range(c_per_group_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = x_g[0, :, i*sy:i*sy+kH, j*sx:j*sx+kW]
                        out[0, out_start+oc, i, j] = np.sum(patch * w_g[oc])

    if bias is not None:
        out += bias.astype(np.float32).reshape(1, -1, 1, 1)


    return _saturate_fp16(out)


def _saturate_fp16(arr):
    'Legacy host helper; see README for precision and scope limits.'
    FP16_MAX = np.float32(65504.0)
    return np.clip(arr, -FP16_MAX, FP16_MAX).astype(np.float16)


def _conv2d_fp16_fast(x, weight, bias, stride, padding, groups):
    'Legacy host helper; see README for precision and scope limits.'
    from numpy.lib.stride_tricks import as_strided

    _, C_in, H, W = x.shape
    C_out, C_grp, kH, kW = weight.shape
    sy, sx = stride
    pt, pl, pb, pr = padding

    if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)),
                    mode='constant', constant_values=0)

    _, _, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // sy + 1
    W_out = (W_pad - kW) // sx + 1



    x_f32 = x.astype(np.float32)[0]
    w_f32 = weight.astype(np.float32)

    if kH == 1 and kW == 1 and groups == 1:

        w_mat = w_f32.reshape(C_out, C_in)  # (C_out, C_in)
        x_flat = x_f32.reshape(C_in, H_out * W_out)  # (C_in, H*W)
        result = w_mat @ x_flat  # (C_out, H*W)
        out = result.reshape(1, C_out, H_out, W_out)

    elif groups == C_in and groups == C_out:

        sh = x_f32.strides
        patches = as_strided(
            x_f32,
            shape=(C_in, H_out, W_out, kH, kW),
            strides=(sh[0], sh[1] * sy, sh[2] * sx, sh[1], sh[2])
        )
        k = w_f32[:, 0, :, :]  # (C, kH, kW)
        out = np.einsum('chwnm,cnm->chw', patches, k)
        out = out[np.newaxis]  # (1, C, H_out, W_out)

    elif groups == 1:

        sh = x_f32.strides
        # (C_in, H_out, W_out, kH, kW)
        patches = as_strided(
            x_f32,
            shape=(C_in, H_out, W_out, kH, kW),
            strides=(sh[0], sh[1] * sy, sh[2] * sx, sh[1], sh[2])
        )
        # (H_out*W_out, C_in*kH*kW)
        col = patches.transpose(1, 2, 0, 3, 4).reshape(H_out * W_out, -1)
        w_mat = w_f32.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        result = w_mat @ col.T  # (C_out, H_out*W_out)
        out = result.reshape(1, C_out, H_out, W_out)

    else:

        c_per_group_in = C_in // groups
        c_per_group_out = C_out // groups
        out = np.zeros((1, C_out, H_out, W_out), dtype=np.float32)
        sh = x_f32.strides
        for g in range(groups):
            in_s = g * c_per_group_in
            out_s = g * c_per_group_out
            x_g = x_f32[in_s:in_s + c_per_group_in]
            w_g = w_f32[out_s:out_s + c_per_group_out]
            sh_g = x_g.strides
            patches = as_strided(
                x_g,
                shape=(c_per_group_in, H_out, W_out, kH, kW),
                strides=(sh_g[0], sh_g[1] * sy, sh_g[2] * sx, sh_g[1], sh_g[2])
            )
            col = patches.transpose(1, 2, 0, 3, 4).reshape(H_out * W_out, -1)
            w_mat = w_g.reshape(c_per_group_out, -1)
            result = w_mat @ col.T
            out[0, out_s:out_s + c_per_group_out] = result.reshape(c_per_group_out, H_out, W_out)

    if bias is not None:
        out += bias.astype(np.float32).reshape(1, -1, 1, 1)



    return np.clip(out, -65504, 65504).astype(np.float16)


class EspressoInference:
    'Legacy host helper; see README for precision and scope limits.'

    def __init__(self, net_path, weights_path):
        self.parser = EspressoParser(net_path, weights_path)
        self._load_weights()

    def _load_weights(self):
        'Legacy host helper; see README for precision and scope limits.'
        self.weights = {}
        for layer in self.parser.layers:
            name = layer['name']
            lt = layer['type']

            if lt == 'convolution':
                K, C = layer['K'], layer['C']
                Nx, Ny = layer.get('Nx', 1), layer.get('Ny', 1)
                ng = layer.get('n_groups', 1)
                w = self.parser.read_blob_f16(layer['blob_weights_f16'])
                self.weights[f'{name}_W'] = w.reshape(C, K // ng, Ny, Nx)
                if layer.get('has_biases') and layer.get('blob_biases') is not None:
                    b = self.parser.read_blob_f16(layer['blob_biases'], as_f16=False)
                    self.weights[f'{name}_B'] = b

            elif lt == 'inner_product':
                nB, nC = layer.get('nB', 0), layer.get('nC', 0)
                w = self.parser.read_blob_f16(layer['blob_weights_f16'])
                self.weights[f'{name}_W'] = w.reshape(nC, nB)
                if layer.get('has_biases') and layer.get('blob_biases') is not None:
                    b = self.parser.read_blob_f16(layer['blob_biases'], as_f16=False)
                    self.weights[f'{name}_B'] = b

    def run(self, input_data, output_layers=None, verbose=False, input_name=None):
        'Legacy host helper; see README for precision and scope limits.'
        output_layers = output_layers or ['Gemm_419']
        key = input_name or 'data'
        tensors = {key: input_data.astype(np.float16)}
        outputs = {}

        total = len(self.parser.layers)
        for i, layer in enumerate(self.parser.layers):
            name = layer['name']
            lt = layer['type']
            bottom = layer.get('bottom', '')
            top = layer.get('top', name)

            if lt == 'convolution':
                self._exec_conv(tensors, layer, name, bottom, top)
            elif lt == 'inner_product':
                self._exec_fc(tensors, layer, name, bottom, top)
            elif lt == 'pool':
                self._exec_pool(tensors, layer, bottom, top)
            elif lt == 'activation':
                self._exec_activation(tensors, layer, bottom, top)
            elif lt == 'elementwise':
                self._exec_elementwise(tensors, layer, bottom, top)
            elif lt == 'softmax':
                self._exec_softmax(tensors, bottom, top)
            elif lt == 'slice':
                self._exec_slice(tensors, layer, bottom, top)
            elif lt == 'concat':
                self._exec_concat(tensors, layer, bottom, top)
            elif lt == 'reshape':
                self._exec_reshape(tensors, layer, bottom, top)
            elif lt == 'transpose':
                self._exec_transpose(tensors, layer, bottom, top)
            elif lt == 'upsample':
                self._exec_upsample(tensors, layer, bottom, top)
            elif lt == 'load_constant':
                self._exec_load_constant(tensors, layer, top)
            elif lt == 'non_maximum_suppression':

                tops = [t.strip() for t in top.split(',')] if top else []
                for t in tops:
                    tensors[t] = None
                if verbose:
                    print(f"  [{i+1}/{total}] (skip NMS) tops={tops}")
            elif lt == 'reduce':

                tensors[top] = None
                if verbose:
                    print(f"  [{i+1}/{total}] (skip reduce) top={top}")
            elif lt == 'flatten':
                self._exec_flatten(tensors, layer, bottom, top)
            elif lt == 'crop':
                self._exec_crop(tensors, layer, bottom, top)
            elif lt == 'deconvolution':
                self._exec_deconv(tensors, layer, bottom, top)
            elif lt == 'loopkit':

                tops = [t.strip() for t in top.split(',')] if top else []
                for t in tops:
                    tensors[t] = None
                if verbose:
                    print(f"  [{i+1}/{total}] (skip loopkit) func={layer.get('func_name')!r} tops={tops}")

            if verbose and (i % 50 == 0 or top in output_layers):
                t = tensors.get(top)
                if t is None:
                    print(f"  [{i+1}/{total}] {top}: <None>")
                else:
                    maxv = np.nanmax(np.abs(t.astype(np.float32)))
                    print(f"  [{i+1}/{total}] {top}: shape={t.shape} max_abs={maxv:.2f} "
                          f"nan={np.isnan(t).sum()} inf={np.isinf(t).sum()}")

            if top in output_layers:
                t = tensors.get(top)
                outputs[top] = None if t is None else t.astype(np.float32)

            if len(outputs) == len(output_layers):
                break

        return outputs

    def _exec_conv(self, tensors, layer, name, bottom, top):
        x = tensors[bottom]
        w = self.weights[f'{name}_W']
        b = self.weights.get(f'{name}_B')

        sy = layer.get('stride_y', 1)
        sx = layer.get('stride_x', 1)
        pt = layer.get('pad_t', 0)
        pb = layer.get('pad_b', 0)
        pl = layer.get('pad_l', 0)
        pr = layer.get('pad_r', 0)
        ng = layer.get('n_groups', 1)
        fused_relu = bool(layer.get('fused_relu', 0))

        out = _conv2d_fp16_fast(x, w, b, (sy, sx), (pt, pl, pb, pr), ng)

        if fused_relu:
            out = np.maximum(out, np.float16(0))

        tensors[top] = out

    def _exec_fc(self, tensors, layer, name, bottom, top):
        x = tensors[bottom]
        w = self.weights[f'{name}_W']
        b = self.weights.get(f'{name}_B')
        has_relu = bool(layer.get('has_relu', 0))

        # 4D → 2D flatten
        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1)


        out = (x.astype(np.float32) @ w.astype(np.float32).T)
        if b is not None:
            out += b.astype(np.float32)
        out = _saturate_fp16(out)

        if has_relu:
            out = np.maximum(out, np.float16(0))

        tensors[top] = out

    def _exec_pool(self, tensors, layer, bottom, top):
        x = tensors[bottom]
        is_global = bool(layer.get('is_global', 0))
        avg_or_max = layer.get('avg_or_max', 0)

        if is_global:
            if avg_or_max == 0:
                out = _saturate_fp16(np.mean(x.astype(np.float32), axis=(2, 3), keepdims=True))
            else:
                out = np.max(x, axis=(2, 3), keepdims=True)
        else:
            kx = layer.get('size_x', 1)
            ky = layer.get('size_y', 1)
            sx = layer.get('stride_x', 1)
            sy = layer.get('stride_y', 1)
            _, C, H, W = x.shape
            H_out = (H - ky) // sy + 1
            W_out = (W - kx) // sx + 1
            out = np.zeros((1, C, H_out, W_out), dtype=np.float16)
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[:, :, i*sy:i*sy+ky, j*sx:j*sx+kx]
                    if avg_or_max == 0:
                        out[:, :, i, j] = _saturate_fp16(np.mean(patch.astype(np.float32), axis=(2, 3)))
                    else:
                        out[:, :, i, j] = np.max(patch, axis=(2, 3))
        tensors[top] = out

    def _exec_activation(self, tensors, layer, bottom, top):
        x = tensors[bottom]
        mode = layer.get('mode', 0)
        if mode == 0:
            out = np.maximum(x, np.float16(0))
        elif mode == 1:
            out = _saturate_fp16(np.tanh(x.astype(np.float32)))
        elif mode == 3:
            # sigmoid: 1/(1+exp(-x))
            out = _saturate_fp16(1.0 / (1.0 + np.exp(-x.astype(np.float32))))
        else:
            out = x
        tensors[top] = out

    def _exec_elementwise(self, tensors, layer, bottom, top):

        bottoms = [b.strip() for b in bottom.split(',')]
        operation = layer.get('operation', 0)
        alpha = float(layer.get('alpha', 1))
        beta = float(layer.get('beta', 0))

        a = tensors[bottoms[0]].astype(np.float32)
        if len(bottoms) == 1:
            base = a
        else:
            b_tensor = tensors[bottoms[1]].astype(np.float32)
            if operation == 0:
                base = a + b_tensor
            elif operation == 1:
                base = a * b_tensor
            elif operation == 2:
                base = np.maximum(a, b_tensor)
            else:
                base = a + b_tensor

        if alpha != 1.0:
            base = base * alpha
        if beta != 0.0:
            base = base + beta

        tensors[top] = _saturate_fp16(base)

    def _exec_softmax(self, tensors, bottom, top):
        x = tensors[bottom].astype(np.float32)
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        tensors[top] = _saturate_fp16(e / np.sum(e, axis=1, keepdims=True))



    def _exec_slice(self, tensors, layer, bottom, top):
        # Espresso axis → numpy 4D NCHW: onnx_axis = 3 - axis (verified in espresso_converter)
        x = tensors[bottom]
        axis = layer.get('axis', 0)
        start = layer.get('start', 0)
        end = layer.get('end', 0)
        np_axis = 3 - axis
        sl = [slice(None)] * x.ndim
        sl[np_axis] = slice(start, end)
        tensors[top] = x[tuple(sl)].copy()

    def _exec_concat(self, tensors, layer, bottom, top):

        bottoms = [b.strip() for b in bottom.split(',')]
        axis = 3 if layer.get('nd_mode', False) else 1
        arrs = [tensors[b] for b in bottoms]
        tensors[top] = np.concatenate(arrs, axis=axis)

    def _exec_reshape(self, tensors, layer, bottom, top):

        x = tensors[bottom]
        dst_n = layer.get('dst_n', 0)
        dst_k = layer.get('dst_k', 0)
        dst_h = layer.get('dst_h', 0)
        dst_w = layer.get('dst_w', 0)
        rank = layer.get('dst_nd_rank', 4)

        n_val = 1 if rank <= 3 else dst_n
        k_val = 1 if rank <= 2 else dst_k
        h_val = dst_h
        w_val = dst_w


        in_shape = list(x.shape)
        if x.ndim != 4:

            in_shape = [1] * (4 - x.ndim) + in_shape

        def resolve(val, idx):
            if val == 0:
                return in_shape[idx]
            return val

        target = [resolve(n_val, 0), resolve(k_val, 1), resolve(h_val, 2), resolve(w_val, 3)]
        tensors[top] = x.reshape(target)

    def _exec_transpose(self, tensors, layer, bottom, top):

        x = tensors[bottom]
        ah = layer.get('axis_h', 0)
        ak = layer.get('axis_k', 1)
        aw = layer.get('axis_w', 2)
        an = layer.get('axis_n', 3)
        if (ah, ak, aw, an) == (0, 1, 2, 3):
            perm = [0, 2, 3, 1]
        else:
            raise ValueError(f"Unsupported transpose axes: h={ah} k={ak} w={aw} n={an}")
        tensors[top] = x.transpose(perm).copy()

    def _exec_upsample(self, tensors, layer, bottom, top):
        # 2x nearest neighbor (mode=0 verified default)
        x = tensors[bottom]
        sx = int(layer.get('scaling_factor_x', 2))
        sy = int(layer.get('scaling_factor_y', 2))
        mode = layer.get('mode', 0)
        if mode != 0:
            raise ValueError(f"Unsupported upsample mode: {mode}")
        # nearest neighbor via repeat: (N, C, H, W) → (N, C, H*sy, W*sx)
        out = np.repeat(np.repeat(x, sy, axis=2), sx, axis=3)
        tensors[top] = out

    def _exec_flatten(self, tensors, layer, bottom, top):
        """flatten: NCHW → (N, C*H*W). nd_axis 0=channels onwards flatten."""
        x = tensors[bottom]
        if x is None:
            tensors[top] = None
            return
        n = x.shape[0]
        tensors[top] = x.reshape(n, -1)

    def _exec_crop(self, tensors, layer, bottom, top):
        """crop: trim spatial dims by l/r/t/b border pixels.
        Layer params: l (left), r (right), t (top), b (bottom)."""
        x = tensors[bottom]
        if x is None:
            tensors[top] = None
            return
        l = int(layer.get('l', 0))
        r = int(layer.get('r', 0))
        t = int(layer.get('t', 0))
        b = int(layer.get('b', 0))
        # Assume NCHW
        H, W = x.shape[-2], x.shape[-1]
        tensors[top] = x[..., t:H - b if b > 0 else H, l:W - r if r > 0 else W].copy()

    def _exec_deconv(self, tensors, layer, bottom, top):
        """Transposed convolution (deconvolution).
        For upsp case: K=245, C=245, n_groups=245 (depth-wise), Nx=Ny=4, stride=2,
        no bias, no BN, no activation.
        Implementation: depth-wise transposed conv via numpy stride tricks.
        """
        x = tensors[bottom]
        if x is None:
            tensors[top] = None
            return
        K = layer['K']  # output channels
        C = layer['C']  # input channels
        Nx = layer['Nx']  # kernel width
        Ny = layer['Ny']  # kernel height
        sx = int(layer.get('stride_x', 1))
        sy = int(layer.get('stride_y', 1))
        groups = layer.get('n_groups', 1)
        pad_t = int(layer.get('pad_t', 0))
        pad_b = int(layer.get('pad_b', 0))
        pad_l = int(layer.get('pad_l', 0))
        pad_r = int(layer.get('pad_r', 0))
        has_bias = bool(layer.get('has_biases', 0))
        fused_relu = bool(layer.get('fused_relu', 0))

        # Load weights
        wblob = layer.get('blob_weights_f16') or layer.get('blob_weights')
        offset, size, _ = self.parser.blob_map[wblob]
        # Weight layout: (K, C/groups, Ny, Nx) for transposed conv
        n_w = K * (C // groups) * Ny * Nx
        if size == n_w * 2:
            w = self.parser.read_blob_f16(wblob, as_f16=True)
        else:
            w = self.parser.read_blob_f16(wblob, as_f16=False)
        w = w.reshape(K, C // groups, Ny, Nx).astype(np.float16)

        N, _, Hin, Win = x.shape
        Hout = (Hin - 1) * sy + Ny - pad_t - pad_b
        Wout = (Win - 1) * sx + Nx - pad_l - pad_r

        # Depth-wise case (most common in upsp)
        if groups == K == C:
            x16 = x.astype(np.float16)
            # Output buffer (un-padded)
            Hpre = (Hin - 1) * sy + Ny
            Wpre = (Win - 1) * sx + Nx
            out = np.zeros((N, K, Hpre, Wpre), dtype=np.float16)
            # Each input pixel scattered with kernel
            for i in range(Hin):
                for j in range(Win):
                    # x16[:, :, i, j] shape (N, K)
                    val = x16[:, :, i, j][:, :, None, None]  # (N, K, 1, 1)
                    out[:, :, i*sy:i*sy+Ny, j*sx:j*sx+Nx] += val * w[:, 0, :, :][None]
            # Crop pad
            out = out[:, :, pad_t:Hpre - pad_b if pad_b > 0 else Hpre,
                            pad_l:Wpre - pad_r if pad_r > 0 else Wpre]
        else:
            raise NotImplementedError(f"Non-depthwise deconv: groups={groups} K={K} C={C}")

        if has_bias:
            bblob = layer['blob_biases']
            b_arr = self.parser.read_blob_f16(bblob, as_f16=True).astype(np.float16)
            out = out + b_arr.reshape(1, K, 1, 1)
        if fused_relu:
            out = np.maximum(out, 0)
        tensors[top] = _saturate_fp16(out.astype(np.float16))

    def _exec_load_constant(self, tensors, layer, top):

        blob_idx = layer['constant_blob']
        n = layer.get('n', 1)
        k = layer.get('k', 1)
        h = layer.get('h', 1)
        w = layer.get('w', 1)
        offset, size, _ = self.parser.blob_map[blob_idx]
        expected = max(n * k * h * w, 1)
        is_f16 = (size == expected * 2)
        if is_f16:
            arr = self.parser.read_blob_f16(blob_idx, as_f16=True)
        else:

            arr = self.parser.read_blob_f16(blob_idx, as_f16=False)
        tensors[top] = arr.reshape(n, k, h, w)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Legacy host inference; not native arithmetic')
    parser.add_argument('--net', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Explicit input .npy array')
    parser.add_argument('--outputs', nargs='+', required=True)
    args = parser.parse_args()
    engine = EspressoInference(args.net, args.weights)
    outputs = engine.run(np.load(args.input, allow_pickle=False), output_layers=args.outputs, verbose=True)
    for name, value in outputs.items():
        print(name, None if value is None else value.shape)


if __name__ == '__main__':
    main()
