"""
Unified array backend: NumPy (default) or PyTorch (when --gpu).
Set via set_backend(use_gpu) before any src code that creates arrays.
"""
from __future__ import annotations
import math
import warnings
from typing import Any

import numpy as np

_BACKEND: str = "numpy"
_DEVICE: Any = None


def set_backend(use_gpu: bool) -> None:
    global _BACKEND, _DEVICE
    if use_gpu:
        try:
            import torch
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                warnings.warn("--gpu requested but CUDA not available; using PyTorch on CPU.")
            _BACKEND = "torch"
        except ImportError:
            warnings.warn("--gpu requested but torch not installed; using NumPy.")
            _BACKEND = "numpy"
            _DEVICE = None
    else:
        _BACKEND = "numpy"
        _DEVICE = None


def get_backend() -> str:
    return _BACKEND


def _torch():
    import torch
    return torch


def _device():
    return _DEVICE


def to_numpy(x: Any) -> np.ndarray:
    """Convert backend array to numpy for plotting/saving."""
    if x is None:
        return x
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def asarray(x: Any, dtype: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.asarray(x, dtype=dtype)
    t = _torch()
    if hasattr(x, "cpu"):  # already tensor
        if dtype is not None:
            dt = t.complex128 if dtype is complex else dtype
            return x.to(_DEVICE).to(dt)
        return x.to(_DEVICE)
    if isinstance(x, (int, float, complex)):
        dt = dtype or (t.complex128 if isinstance(x, complex) else t.float64)
        if dt is complex:
            dt = t.complex128
        return t.tensor(x, dtype=dt, device=_DEVICE)
    a = np.asarray(x, dtype=dtype)
    return t.from_numpy(a).to(_DEVICE)


def array(x: Any, dtype: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.array(x, dtype=dtype)
    if isinstance(x, (list, tuple)) and x and hasattr(x[0], "cpu"):
        return _torch().stack(x)
    return asarray(x, dtype=dtype)


def stack(arrays: list, axis: int = 0) -> Any:
    if _BACKEND == "numpy":
        return np.stack(arrays, axis=axis)
    return _torch().stack(arrays, dim=axis)


def zeros(shape: tuple[int, ...], dtype: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.zeros(shape, dtype=dtype or float)
    t = _torch()
    dt = dtype or t.float64
    if isinstance(dt, type) and dt == complex:
        dt = t.complex128
    return t.zeros(shape, dtype=dt, device=_DEVICE)


def eye(n: int, dtype: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.eye(n, dtype=dtype or float)
    t = _torch()
    dt = dtype or t.float64
    if isinstance(dt, type) and dt == complex:
        dt = t.complex128
    return t.eye(n, dtype=dt, device=_DEVICE)


def inv(A: Any) -> Any:
    if _BACKEND == "numpy":
        return np.linalg.inv(A)
    return _torch().linalg.inv(A)


def eigvalsh(A: Any) -> Any:
    """Real eigenvalues of Hermitian matrix."""
    if _BACKEND == "numpy":
        return np.linalg.eigvalsh(A).real
    return _torch().linalg.eigvalsh(A).real


def norm(x: Any, ord: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.linalg.norm(x, ord=ord)
    return _torch().linalg.norm(x, ord=ord)


def exp(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.exp(x)
    t = _torch()
    if isinstance(x, (int, float, complex)):
        dt = t.complex128 if isinstance(x, complex) else t.float64
        return t.tensor(x, dtype=dt, device=_DEVICE).exp()
    return t.exp(x)


def dot(a: Any, b: Any) -> Any:
    if _BACKEND == "numpy":
        return np.dot(a, b)
    if a.ndim == 1 and b.ndim == 1:
        return _torch().dot(a, b)
    return _torch().matmul(a, b)


def kron(A: Any, B: Any) -> Any:
    if _BACKEND == "numpy":
        return np.kron(A, B)
    return _torch().kron(A, B)


def conj(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.conj(x)
    return _torch().conj(x)


def T(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.transpose(x)
    return x.T


def H(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.conj(np.transpose(x))
    return _torch().conj(x.T)


def trace(x: Any, axis1: int = 0, axis2: int = 1) -> Any:
    if _BACKEND == "numpy":
        return np.trace(x, axis1=axis1, axis2=axis2)
    t = _torch()
    if x.dim() == 2:
        return t.trace(x)
    return t.diagonal(x, dim1=axis1, dim2=axis2).sum(-1)


def real(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.real(x)
    return _torch().real(x)


def imag(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.imag(x)
    return _torch().imag(x)


def abs(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.abs(x)
    return _torch().abs(x)


def maximum(a: Any, b: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.maximum(a, b) if b is not None else np.max(a)
    t = _torch()
    if b is not None:
        return t.maximum(a, b)
    return a.max()


def minimum(a: Any, b: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.minimum(a, b) if b is not None else np.min(a)
    t = _torch()
    if b is not None:
        return t.minimum(a, b)
    return a.min()


def amax(x: Any, *args, **kwargs) -> Any:
    if _BACKEND == "numpy":
        return np.max(x, *args, **kwargs)
    return _torch().amax(x, *args, **kwargs)


def amin(x: Any, *args, **kwargs) -> Any:
    if _BACKEND == "numpy":
        return np.min(x, *args, **kwargs)
    return _torch().amin(x, *args, **kwargs)


def sum(x: Any, *args, **kwargs) -> Any:
    if _BACKEND == "numpy":
        return np.sum(x, *args, **kwargs)
    return _torch().sum(x, *args, **kwargs)


def mean(x: Any, *args, **kwargs) -> Any:
    if _BACKEND == "numpy":
        return np.mean(x, *args, **kwargs)
    return _torch().mean(x, *args, **kwargs)


def cross(a: Any, b: Any, axisa: int = -1, axisb: int = -1, axisc: int = -1) -> Any:
    if _BACKEND == "numpy":
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc)
    t = _torch()
    if a.dim() == 1:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        out = t.linalg.cross(a, b, dim=-1).squeeze(0)
    else:
        out = t.linalg.cross(a, b, dim=-1)
    return out


def copy(x: Any) -> Any:
    if _BACKEND == "numpy":
        return np.copy(x)
    return x.clone()


def linspace(start: float, stop: float, num: int, dtype: Any = None) -> Any:
    if _BACKEND == "numpy":
        return np.linspace(start, stop, num, dtype=dtype)
    t = _torch()
    return t.linspace(start, stop, num, dtype=dtype or t.float64, device=_DEVICE)


def sqrt(x: float) -> float:
    return math.sqrt(x)


def diag(v: Any) -> Any:
    """Create diagonal matrix from 1D array or list."""
    if _BACKEND == "numpy":
        return np.diag(np.asarray(v, dtype=float))
    t = _torch()
    if hasattr(v, "cpu"):
        v = v.to(_DEVICE)
    else:
        v = t.tensor(v, dtype=t.float64, device=_DEVICE)
    return t.diag(v)
