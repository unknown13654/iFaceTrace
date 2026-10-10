"""NumPy archive output with only ZIP creator-system metadata fixed to Unix."""
import zipfile

import numpy as np


def write_npz(path, arrays):
    with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in arrays.items():
            entry = zipfile.ZipInfo(name + '.npy')
            entry.create_system = 3
            with archive.open(entry, mode='w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)
