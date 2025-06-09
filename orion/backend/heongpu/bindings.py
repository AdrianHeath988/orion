import os 
import ctypes
import platform

import torch
import numpy as np

class HEonGPUFunction:
    """Helper to wrap ctypes functions with argument and return types."""
    def __init__(self, func, argtypes, restype):
        self.func = func
        self.func.argtypes = argtypes 
        self.func.restype = restype

    def __call__(self, *args):
        c_args = []
        for arg in args:
            curr_argtype = self.func.argtypes[len(c_args)]
            c_arg = self.convert_to_ctypes(arg, curr_argtype)
            if isinstance(c_arg, tuple):
                c_args.extend(c_arg)
            else:
                c_args.append(c_arg)
                
        c_result = self.func(*c_args)
        py_result = self.convert_from_ctypes(c_result)
        
        # If the result is a list, then we'll need to manually free the
        # memory we allocated for this list in C with the below. We'll
        # defer freeing byte data (from serialization) until after that
        # data has been saved to HDF5.
        if isinstance(py_result, list):
            HEonGPUFunction.FreeCArray(
                ctypes.cast(c_result.Data, ctypes.c_void_p))

        return py_result

    @torch._dynamo.disable
    def convert_to_ctypes(self, arg, typ):
        if isinstance(arg, int) and typ == ctypes.c_int:
            return ctypes.c_int(arg)
        elif isinstance(arg, int) and typ == ctypes.c_ulong:
            return ctypes.c_ulong(arg)
        elif isinstance(arg, float):
            return ctypes.c_float(arg)
        elif isinstance(arg, str):
            return arg.encode('utf-8')
        elif (isinstance(arg, np.ndarray) and 
            arg.dtype == np.uint8 and 
            typ == ctypes.POINTER(ctypes.c_ubyte)):
            ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            return (ptr, len(arg))
        elif isinstance(arg, list):
            if typ == ctypes.POINTER(ctypes.c_int):
                return ((ctypes.c_int * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_float):
                return ((ctypes.c_float * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ulong):
                return ((ctypes.c_ulong * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ubyte):
                return ((ctypes.c_ubyte * len(arg))(*arg), len(arg))
            else:
                raise ValueError("Unexpected list type to convert.")
        else:
            return arg
            
    def convert_from_ctypes(self, res):
        if type(res) == ctypes.c_int:
            return int(res)
        elif type(res) == ctypes.c_float:
            return float(res)
        elif type(res) == ArrayResultFloat:
            return [float(res.Data[i]) for i in range(res.Length)]
        elif type(res) in (ArrayResultInt, ArrayResultUInt64):
            return [int(res.Data[i]) for i in range(res.Length)]
        elif type(res) == ArrayResultDouble:
            return [float(res.Data[i]) for i in range(res.Length)]
        elif type(res) == ArrayResultByte:
            # Create numpy array directly from the C buffer
            buffer = ctypes.cast(
                res.Data, 
                ctypes.POINTER(ctypes.c_ubyte * res.Length)
            ).contents
            array = np.frombuffer(buffer, dtype=np.uint8)
            return array, res.Data
        else:
            return res


class HEonGPULibrary:
    """A class to manage loading and interfacing with Lattigo."""
    def __init__(self):
        self.lib = self._load_library()

    def _load_library(self):
        try:
            lib_name = "libheongpu_c_api-1.1.so.1.1" # direct loading
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            
            relative_build_path = 'adrianHEonGPU/build_heongpu/src/heongpu/'
            lib_path = os.path.join(current_dir, relative_build_path, lib_name)
            
            print(f"Attempting to load library from: {lib_path}")
            
            # Check if the file exists before trying to load it
            if not os.path.exists(lib_path):
                # Fallback: maybe the library is in the same directory as the script
                lib_path = os.path.join(current_dir, lib_name)
                if not os.path.exists(lib_path):
                    raise RuntimeError(f"Library file not found at expected path: {lib_name}")

            return ctypes.CDLL(lib_path)
            
        except OSError as e:
            raise RuntimeError(f"Failed to load HEonGPU C API library: {e}")
    
    def _find_library(self, root_dir, lib_name):
        """Recursively search for the library file"""
        for root, _, files in os.walk(root_dir):
            if lib_name in files:
                return os.path.join(root, lib_name)
        raise FileNotFoundError(f"Library {lib_name} not found in {root_dir}")

    def setup_bindings(self, orion_params):
        """
        Declares the functions from the Lattigo shared library and sets their
        argument and return types.
        """
        # self.setup_scheme(orion_params)
        # self.setup_tensor_binds()
        # self.setup_key_generator()
        # self.setup_encoder()
        # self.setup_encryptor()
        # self.setup_evaluator()
        # self.setup_poly_evaluator()
        # self.setup_lt_evaluator()
        # self.setup_bootstrapper()
