import h5py
from pathlib import Path
import numpy as np


def ivecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12.
    The loaded dataset should be in the format described here: http://corpus-texmex.irisa.fr/
    """
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12
    The loaded dataset should be in the format described here: http://corpus-texmex.irisa.fr/
    """
    return ivecs_read(filename).view('float32')

def parse_ann_benchmark(filename: str | Path) -> (np.ndarray, np.ndarray):
    """
    Parses hdf5 file similar to ann-benchmark.com. Ground truth is loaded separately
    """
    with h5py.File(filename, 'r') as file:
        return np.array(file['train']), np.array(file['test'])

def get_rolling_update_gt(filename: str | Path) -> np.ndarray:
    """
    Parses hdf5 rolling update ground truth file, similar format to ann-benchmark.com
    """
    with h5py.File(filename, 'r') as file:
        return np.array(file["neighbors"])