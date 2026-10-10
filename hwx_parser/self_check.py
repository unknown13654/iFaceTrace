"""Small synthetic boundary checks; no Apple files or hardware are required."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from espresso_inference import EspressoParser
from freeze_inputs import copy_input
from inspect_hwx import kernel_file_offset
from sparse_decode import self_check as check_sparse
from verify_lane_layout import decode_dense
from npz_io import write_npz


class Boundaries(unittest.TestCase):
    def test_npz_creator_metadata_preserves_arrays_order_dates_and_other_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {'second': np.array([1., -0.], dtype='<f2'), 'first': np.array([[3., 4.]], dtype='<f2')}
            raw, fixed = root / 'raw.npz', root / 'fixed.npz'
            np.savez(raw, **arrays)
            write_npz(fixed, arrays)
            raw_bytes, fixed_bytes = raw.read_bytes(), fixed.read_bytes()
            self.assertEqual(len(raw_bytes), len(fixed_bytes))
            with zipfile.ZipFile(raw) as old, zipfile.ZipFile(fixed) as new:
                self.assertEqual(old.namelist(), new.namelist())
                position = new.start_dir
                expected = bytearray(raw_bytes)
                for entry, previous in zip(new.infolist(), old.infolist()):
                    self.assertEqual(entry.create_system, 3)
                    self.assertEqual(entry.date_time, previous.date_time)
                    self.assertEqual(new.read(entry), old.read(previous))
                    expected[position + 5] = 3
                    position += 46 + sum(struct.unpack_from('<3H', fixed_bytes, position + 28))
                self.assertEqual(bytes(expected), fixed_bytes)
            with np.load(fixed, allow_pickle=False) as loaded:
                self.assertEqual(loaded.files, list(arrays))
                for name, array in arrays.items():
                    self.assertEqual(loaded[name].dtype, array.dtype)
                    self.assertEqual(loaded[name].shape, array.shape)
                    self.assertEqual(loaded[name].tobytes(), array.tobytes())

    def test_dense_lane_order_and_bias(self):
        data = np.array([2, 3, 10, 11, 20, 21], dtype='<f2').tobytes()
        weights, bias = decode_dense(data, [0], 2, 2, 2)
        self.assertEqual(weights.tolist(), [[10, 20], [11, 21]])
        self.assertEqual(bias.tolist(), [2, 3])

    def test_dense_prefix_and_signed_zero_bits(self):
        data = np.array([-0., 1., -0., 2., 3., 4.], dtype='<f2').tobytes()
        weights, bias = decode_dense(bytes(128) + data, [0], 2, 2, 2, prefix_bytes=128)
        self.assertTrue(np.signbit(bias[0]))
        self.assertTrue(np.signbit(weights[0, 0]))

    def test_dense_short_input_is_rejected(self):
        with self.assertRaises(AssertionError):
            decode_dense(bytes(2), [0], 2, 2, 2)

    def test_sparse_stream_self_check(self):
        check_sparse()

    def test_section_end_marker_is_not_a_buffer(self):
        section = {'address': 100, 'size': 10, 'fileoff': 20}
        self.assertEqual(kernel_file_offset(section, 109), 29)
        self.assertIsNone(kernel_file_offset(section, 110))
        with self.assertRaises(AssertionError):
            kernel_file_offset(section, 111)

    def test_original_blob_table_and_copy_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net, weights = root / 'model.net', root / 'model.weights'
            net.write_text('{"layers":[]}', encoding='utf-8')
            payload = np.array([1., -0.], dtype='<f2').tobytes()
            weights.write_bytes(struct.pack('<QQQ', 1, 7, len(payload)) + payload)
            parser = EspressoParser(net, weights)
            self.assertEqual(parser.blob_map[7][:2], (24, 4))
            copied = root / 'copy.weights'
            receipt = copy_input(weights, copied)
            self.assertEqual(receipt['sha256'], hashlib.sha256(weights.read_bytes()).hexdigest())
            self.assertEqual(copied.read_bytes(), weights.read_bytes())
            with self.assertRaises(AssertionError):
                copy_input(weights, copied)

    def test_unknown_input_hash_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / 'unknown.bin'
            unknown.write_bytes(b'not the declared Apple resource')
            output = root / 'output'
            argv = [sys.executable, '-B', str(Path(__file__).with_name('export_face_tensors.py'))]
            for name in ('net', 'shape', 'hwx', 'compiler', 'mapping', 'lane-layout', 'sparse-results', 'sigmoid-prefix'):
                argv.extend(['--' + name, str(unknown)])
            argv.extend(['--ipsw', 'ipsw', '--output-dir', str(output)])
            done = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn('input SHA-256 mismatch', done.stderr)
            self.assertFalse(output.exists())

    def test_missing_explicit_arguments_fail_closed(self):
        done = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('export_face_tensors.py'))], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn('required', done.stderr)


if __name__ == '__main__':
    unittest.main()
