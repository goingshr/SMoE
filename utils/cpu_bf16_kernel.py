"""Optional AVX2 BF16 GEMV. Compilation is explicit and cached by PyTorch."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import threading

import torch

_lock = threading.Lock()
_loaded = False


def load_kernel():
    global _loaded
    with _lock:
        if not _loaded:
            from torch.utils.cpp_extension import include_paths, library_paths
            source = Path(__file__).with_name('cpu_bf16_gemv.cpp')
            flags = ['-O3', '-mavx2', '-mfma', '-fopenmp', '-std=c++17', '-shared', '-fPIC']
            key = hashlib.sha256(source.read_bytes() + (torch.__version__+str(flags)).encode()).hexdigest()[:16]
            cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', tempfile.gettempdir())) / 'smoe_bf16'
            cache.mkdir(parents=True, exist_ok=True)
            binary = cache / f'gemv_{key}.so'
            if not binary.exists():
                tmp = binary.with_suffix(f'.{os.getpid()}.so')
                cmd = [os.environ.get('CXX', 'g++'), *flags,
                    f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
                    *[f'-I{x}' for x in include_paths()], str(source), '-o', str(tmp),
                    *[f'-L{x}' for x in library_paths()], '-ltorch_cpu', '-ltorch', '-lc10',
                    *[f'-Wl,-rpath,{x}' for x in library_paths()]]
                subprocess.run(cmd, check=True)
                tmp.replace(binary)
            torch.ops.load_library(str(binary))
            _loaded = True
    return torch.ops.smoe_cpu.bf16_gemv
