import os 
import ctypes
import platform
import math
from pathlib import Path
import torch
import numpy as np
import hashlib
from orion.backend.python.tensors import CipherTensor

class HE_CKKS_Context(ctypes.Structure): pass
class HE_CKKS_Plaintext(ctypes.Structure): pass
class HE_CKKS_Ciphertext(ctypes.Structure): pass
class HE_CKKS_SecretKey(ctypes.Structure): pass
class HE_CKKS_PublicKey(ctypes.Structure): pass
class HE_CKKS_MultipartyPublicKey(ctypes.Structure): pass
class HE_CKKS_RelinKey(ctypes.Structure): pass
class HE_CKKS_MultipartyRelinKey(ctypes.Structure): pass
class HE_CKKS_GaloisKey(ctypes.Structure): pass
class HE_CKKS_Encoder(ctypes.Structure): pass
class HE_CKKS_Encryptor(ctypes.Structure): pass
class HE_CKKS_Decryptor(ctypes.Structure): pass
class HE_CKKS_KeyGenerator(ctypes.Structure): pass
class HE_CKKS_ArithmeticOperator(ctypes.Structure): pass
class HE_CKKS_LogicOperator(ctypes.Structure): pass
C_storage_type = ctypes.c_int
class C_ExecutionOptions(ctypes.Structure):
    _fields_ = [
        ("stream", ctypes.c_void_p),
        ("storage", C_storage_type),
        ("keep_initial_condition", ctypes.c_bool)
    ]

class C_BootstrappingConfig(ctypes.Structure):
    _fields_ = [
        ("CtoS_piece", ctypes.c_int),
        ("StoC_piece", ctypes.c_int),
        ("taylor_number", ctypes.c_int),
        ("less_key_mode", ctypes.c_bool)
    ]

class C_Modulus64(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_uint64),
        ("bit", ctypes.c_int),
        ("mu", ctypes.c_uint64) #assuming mu is uint64_t
    ]

class ArrayResultInt(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_int)), 
                ("Length", ctypes.c_size_t)]

class ArrayResultFloat(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_float)), 
                ("Length", ctypes.c_size_t)]

class ArrayResultDouble(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_double)), ("Length", ctypes.c_ulong)]

class ArrayResultUInt64(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_uint64)), # c_uint64 is an alias for c_ulonglong
                ("Length", ctypes.c_size_t)]

class ArrayResultByte(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_ubyte)), # (unsigned char *)
                ("Length", ctypes.c_size_t)]
class HE_CKKS_Plaintext(ctypes.Structure):
    _fields_ = [("cpp_plaintext", ctypes.c_void_p)]

_heongpu_instance = None

def get_heongpu_library():
    global _heongpu_instance
    if _heongpu_instance is None:
        _heongpu_instance = HEonGPULibrary()
    return _heongpu_instance


class HEonGPUFunction:
    def __init__(self, func, argtypes=None, restype=None):
        self.func = func
        if argtypes:
            self.func.argtypes = argtypes
        if restype:
            self.func.restype = restype

    def __call__(self, *args):
        # This is the corrected __call__ method, modeled on the Lattigo wrapper.
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        #     allocated_mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
        #     reserved_mem_mb = torch.cuda.memory_reserved(device) / (1024**2)
        #     print(f"Allocated memory: {allocated_mem_mb} MB")
        #     print(f"Reserved memory: {reserved_mem_mb} MB")
        #     free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device)
            
        #     free_mem_mb = free_mem_bytes / (1024**2)
        #     total_mem_mb = total_mem_bytes / (1024**2)
            
        #     print(f"Free memory: {free_mem_mb} MB")
        #     print(f"Total memory: {total_mem_mb} MB")
        
        # print(f"Calling C function: {self.func.__name__}")





        c_args = []
        for arg in args:
            # This is the key: get the current C argtype based on the length
            # of the c_args list, which has already been processed.
            curr_argtype = self.func.argtypes[len(c_args)]
            
            # Call the existing helper function to convert the Python arg.
            c_arg = self.convert_to_ctypes(arg, curr_argtype)
            
            # If the helper returned a (pointer, size) tuple, add both to the list.
            if isinstance(c_arg, tuple):
                c_args.extend(c_arg)
            else:
                # Otherwise, just add the single converted argument.
                c_args.append(c_arg)
                
        # Call the C function with the fully processed list of arguments.
        c_result = self.func(*c_args)

        py_result = self.convert_from_ctypes(c_result, type(c_result))
        return py_result




    @torch._dynamo.disable
    def convert_to_ctypes(self, arg, typ):
        if arg is None:
            return None
        if isinstance(arg, (int, float)):
            if typ == ctypes.c_double:
                return ctypes.c_double(float(arg))
            if typ == ctypes.c_float:
                return ctypes.c_float(float(arg))
            if typ == ctypes.c_int:
                return ctypes.c_int(int(arg))
            if typ == ctypes.c_ulong:
                return ctypes.c_ulong(int(arg))
        elif isinstance(arg, str):
            return arg.encode('utf-8')
        
        elif isinstance(arg, np.ndarray):
            if not arg.flags['C_CONTIGUOUS']:
                arg = np.ascontiguousarray(arg)

            if arg.dtype == np.float64 and typ == ctypes.POINTER(ctypes.c_double):
                ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                return (ptr, len(arg))
            elif arg.dtype == np.uint8 and typ == ctypes.POINTER(ctypes.c_ubyte):
                ptr = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                return (ptr, len(arg))
            else:
                 raise TypeError(f"Unsupported numpy array dtype '{arg.dtype}' for C conversion.")
        
        elif isinstance(arg, list):
            c_elem_type = None
            if typ == ctypes.POINTER(ctypes.c_int): c_elem_type = ctypes.c_int
            elif typ == ctypes.POINTER(ctypes.c_uint64): c_elem_type = ctypes.c_uint64
            elif typ == ctypes.POINTER(ctypes.c_ulong): c_elem_type = ctypes.c_ulong
            elif typ == ctypes.POINTER(ctypes.c_ubyte): c_elem_type = ctypes.c_ubyte
            elif typ == ctypes.POINTER(ctypes.c_float): c_elem_type = ctypes.c_float
            elif typ == ctypes.POINTER(ctypes.c_double): c_elem_type = ctypes.c_double 
            
            if c_elem_type:
                ArrayType = c_elem_type * len(arg)
                return (ArrayType(*arg), len(arg))
            else:
                raise ValueError(f"Unexpected list conversion to pointer type: {typ}")
        else:
            return arg 
            
    def convert_from_ctypes(self, res, restype):
        if (hasattr(restype, '_type_') and isinstance(restype._type_, type) and issubclass(restype._type_, ctypes.Structure)):
            if not res:
                return None
            struct = res.contents
            struct_type = type(struct)
            if struct_type == ArrayResultFloat:
                return [float(struct.Data[i]) for i in range(struct.Length)]
            elif struct_type in (ArrayResultInt, ArrayResultUInt64):
                return [int(struct.Data[i]) for i in range(struct.Length)]
            elif struct_type == ArrayResultDouble:
                if not struct.Data:
                    print("[ERROR] The structure's internal Data pointer is NULL.")
                    return []
                return [float(struct.Data[i]) for i in range(struct.Length)]
            elif struct_type == ArrayResultByte:
                buffer = ctypes.cast(struct.Data, ctypes.POINTER(ctypes.c_ubyte * struct.Length)).contents
                array = np.frombuffer(buffer, dtype=np.uint8)
                return array, struct.Data 
            else:
                return res

        elif restype in (ctypes.c_int, ctypes.c_long, ctypes.c_longlong, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulonglong, ctypes.c_size_t):
            return int(res)
        elif restype in (ctypes.c_float, ctypes.c_double):
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
        self.linear_transforms = []
        self.rotation_keys_cache = {}
        print(f"DEBUG: Successfully loaded library from: {self.lib._name}")

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
        self.context_handle = self.setup_scheme(orion_params)
        self.setup_tensor_binds()
        self.setup_key_generator()
        self.setup_encoder()
        self.setup_encryptor()
        self.setup_evaluator()
        self.setup_poly_evaluator()
        self.setup_lt_evaluator()
        self.setup_bootstrapper()

    def shutdown(self):
        print("Requesting manual HEonGPU resource cleanup...")
        self.Shutdown = HEonGPUFunction(
            self.lib.heongpu_shutdown,
            argtypes=[
            ],
            restype=None
        )
        #Now, call it
        self.Shutdown()
        

    #setup scheme
    def setup_scheme(self, orion_params):
        """
        Initializes and configures the HEonGPU scheme by creating and setting up
        a CKKS context object.

        This function binds to the C API functions defined in context_c_api.h.
        """
        self.HEonGPU_CKKS_SynchronizeDevice = HEonGPUFunction(
            self.lib.HEonGPU_SynchronizeDevice,
            argtypes=[             
            ],
            restype=ctypes.c_int
        )

        self.HEonGPU_CKKS_Context_Create = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_Create,
            argtypes=[
                ctypes.c_int,  # C_keyswitching_type enum
                ctypes.c_int   # C_sec_level_type enum
            ],
            # Returns an opaque pointer to the context struct
            restype=ctypes.POINTER(HE_CKKS_Context) 
        )
        
        self.HEonGPU_CKKS_Context_Delete = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_Delete,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=None
        )
        
        self.HEonGPU_CKKS_Context_SetPolyModulusDegree = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_SetPolyModulusDegree,
            argtypes=[ctypes.POINTER(HE_CKKS_Context), ctypes.c_size_t],
            restype=None
        )
        self.HEonGPU_CKKS_Context_GetPolyModulusDegree = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_GetPolyModulusDegree,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.c_size_t
        )



        self.HEonGPU_CKKS_Context_SetCoeffModulusValues = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_SetCoeffModulusValues,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(ctypes.c_uint64), # log_q_bases_data (uint64_t*)
                ctypes.c_size_t,               # log_q_bases_len
                ctypes.POINTER(ctypes.c_uint64), # log_p_bases_data (uint64_t*)
                ctypes.c_size_t                # log_p_bases_len
            ],
            restype=ctypes.c_int # Returns 0 on success
        )
        
        self.HEonGPU_CKKS_Context_SetCoeffModulusBitSizes = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_SetCoeffModulusBitSizes,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(ctypes.c_int), 
                ctypes.c_size_t,              
                ctypes.POINTER(ctypes.c_int), 
                ctypes.c_size_t                
            ],
            restype=ctypes.c_int 
        )

        self.HEonGPU_CKKS_Context_Generate = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_Generate,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.c_int # Returns 0 on success
        )
        
        self.HEonGPU_FreeSerializedData = HEonGPUFunction(
            self.lib.HEonGPU_FreeSerializedData,
            argtypes=[ctypes.c_void_p],
            restype=None
        )

        self.HEonGPU_PrintParameters = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_PrintParameters,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=None
        )

        
        # HEonGPU parameters (using CKKS defaults where applicable)
        keyswitch_method = 2
        # can set to default (128) or can let HEonGPU choose
        # sec_level = 128
        sec_level = 0
        context_handle = self.HEonGPU_CKKS_Context_Create(keyswitch_method, sec_level)
        if not context_handle:
            raise RuntimeError("Failed to create HEonGPU CKKS Context shell.")

        poly_degree = 1 << orion_params.get_logn()
        self.poly_degree = poly_degree 
        logq = orion_params.get_logq()
        logp = orion_params.get_logp()
        print(orion_params.get_logscale())
        self.scale = 2.0 ** orion_params.get_logscale()
        self.q_size = len(logq)
        print(f"INFO: Setting PolyModulusDegree to {poly_degree}")
        self.HEonGPU_CKKS_Context_SetPolyModulusDegree(context_handle, poly_degree)

        print(f"INFO: Setting CoeffModulus with LogQ: {logq} and LogP: {logp}")
        result_modulus = self.HEonGPU_CKKS_Context_SetCoeffModulusBitSizes(context_handle, logq, logp)
        if result_modulus != 0:
            self.HEonGPU_CKKS_Context_Delete(context_handle)
            raise RuntimeError(f"Failed to set HEonGPU coefficient modulus bit-sizes. Status: {result_modulus}")
        print("INFO: Generating HEonGPU context with specified parameters...")

        result_generate = self.HEonGPU_CKKS_Context_Generate(context_handle)
        
        if result_generate != 0:
            self.HEonGPU_CKKS_Context_Delete(context_handle)
            raise RuntimeError(f"Failed to generate HEonGPU context parameters. Status: {result_generate}")

        cpp_poly_degree = self.HEonGPU_CKKS_Context_GetPolyModulusDegree(context_handle)
        print(f"  Python wrapper's poly_degree: {self.poly_degree}")
        print(f"  Actual C++ context's poly_degree: {cpp_poly_degree}")

        self.HEonGPU_PrintParameters(context_handle)
        print("INFO: HEonGPU CKKS Context successfully created and configured.")
        

        return context_handle


    #tensor binds
    def setup_tensor_binds(self):
        self._DeletePlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_Delete,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
            restype=None
        )
        self._DeleteCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_Delete,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=None
        )

        self.GetPlaintextScale = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_GetScale,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
            restype=ctypes.c_double
        )

        self.GetCiphertextScale = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetScale,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_double
        )

        # Not A thing
        # self.SetPlaintextScale = HEonGPUFunction(
        #     self.lib.HEonGPU_CKKS_Plaintext_SetScale,
        #     argtypes=[
        #         ctypes.POINTER(HE_CKKS_Plaintext),
        #         ctypes.c_double,
        #     ],
        #     restype=None
        # )

        self._SetCiphertextScale = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_Set_Scale,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.c_double,
            ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )

        # "Level" corresponds to the number of remaining prime moduli in the chain.
        # However, plaintext in HEonGPU stores depth (number of prime moduli consumed).
        # Need to think on how to resolvbe this.
        self.GetPlaintextDepth = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_GetDepth,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
            restype=ctypes.c_int
        )

        self.GetCiphertextLevel = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetCoeffModulusCount, # Assumes this C API function exists
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_int
        )

        # "Slots" corresponds to half the ring size (poly_modulus_degree).
        self.GetPlaintextSize = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_GetPlainSize,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
            restype=ctypes.c_int 
        )
        self.GetCiphertextSize = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetRingSize,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_int
        )
        self.GetCiphertextDepth = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetDepth,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_int
        )


        self.GetCiphertextDegree = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetCiphertextSize,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_int
        )

        self._GetModuliChain = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Context_GetCoeffModulus,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(C_Modulus64),
                ctypes.c_size_t
            ],
            restype=ctypes.c_size_t
        )
        self._CopyCiphertext = self.lib.HEonGPU_CKKS_Ciphertext_Assign_Copy
        self._CopyCiphertext.argtypes = [
            ctypes.POINTER(HE_CKKS_Ciphertext), # Destination
            ctypes.POINTER(HE_CKKS_Ciphertext)  # Source
        ]
        self._CopyCiphertext.restype = ctypes.c_int
    
    def CloneCiphertext(self, ctxt_in):
        # Creates a new ciphertext that is a copy of the input.
        ctxt_out = self.CreateCiphertext()
        
        # Call the C++ copy function and check the return status
        status = self._CopyCiphertext(ctxt_out, ctxt_in)
        if status != 0:
            raise RuntimeError(f"HEonGPU_CKKS_Ciphertext_Assign_Copy failed with status code {status}")
        
        return ctxt_out

    def DeleteCiphertext(self, ct):
        if(self.context_handle != None):  #only try to delete if memory pool exists
            self._DeleteCiphertext(ct)
    
    def GetModuliChain(self):
        required_size = self._GetModuliChain(
            self.context_handle,
            None,
            0
        )
        if required_size == 0:
            print("DEBUG: GetModuliChain required_size 0")
            return []
        BufferType = C_Modulus64 * required_size
        moduli_buffer = BufferType()
        num_copied = self._GetModuliChain(
            self.context_handle,
            moduli_buffer,
            required_size
        )
        if num_copied != required_size:
            print(f"Warning: Expected to copy {required_size} moduli, but only got {num_copied}.")

        return [moduli_buffer[i].value for i in range(num_copied)]

    def GetPlaintextSlots(self, plaintext_handle):
        if not plaintext_handle:
            raise ValueError("Invalid plaintext handle provided.")
        plain_size = self.GetPlaintextSize(plaintext_handle)
        return plain_size // 2
    
    def GetCiphertextSlots(self, ciphertext_handle):
        # This function now reflects the "Standard" ring type where slots = N/2
        if hasattr(self, 'poly_degree'):
            return self.poly_degree // 2
        raise ValueError("poly_degree not available.")


    #This function asserts that the current scale is equivilent to the scale it is being 'set' to
    #If not, then an error will eb thrown
    def SetCiphertextScale(self, ct, scale):
        current_scale = self.GetCiphertextScale(ct)
        if(math.isclose(current_scale, scale)):
            return ct
        else:
            #try to rescale:
            ct = self.Rescale(ct)
            current_scale = self.GetCiphertextScale(ct)
            if(math.isclose(current_scale, scale)):
                return ct
            else:
                #All else fails: set scale manually
                return ct
                ct = self._SetCiphertextScale(ct, scale)
                current_scale = self.GetCiphertextScale(ct)
                if(math.isclose(current_scale, scale)):
                    return ct
                else:
                    #Should never reach this point
                    print(f"\n[DEBUG] Current Scale is {current_scale}, requested scale is {scale}.\n")
                    raise ValueError("scales not equal.")

    


    #key generator
    def setup_key_generator(self):
        self._NewKeyGenerator = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_KeyGenerator_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.POINTER(HE_CKKS_KeyGenerator)
        )

        self.CreateSecretKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_SecretKey_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.POINTER(HE_CKKS_SecretKey)
        )

        # Key must already exist
        self._GenerateSecretKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_KeyGenerator_GenerateSecretKey,
            argtypes=[
                ctypes.POINTER(HE_CKKS_KeyGenerator),
                ctypes.POINTER(HE_CKKS_SecretKey),
                ctypes.POINTER(C_ExecutionOptions)   # Pointer to execution options (can be None/null)
            ],
            restype=ctypes.c_int 
        )

        self.CreatePublicKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_PublicKey_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.POINTER(HE_CKKS_PublicKey)
        )
        self._GeneratePublicKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_KeyGenerator_GeneratePublicKey,
            argtypes=[
                ctypes.POINTER(HE_CKKS_KeyGenerator),
                ctypes.POINTER(HE_CKKS_PublicKey),
                ctypes.POINTER(HE_CKKS_SecretKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.c_int 
        )

        self.CreateRelinearizationKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_RelinKey_Create,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.c_bool   #store_in_gpu
            ],
            restype=ctypes.POINTER(HE_CKKS_RelinKey)
        )
        self._GenerateRelinearizationKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_KeyGenerator_GenerateRelinKey,
            argtypes=[
                ctypes.POINTER(HE_CKKS_KeyGenerator),
                ctypes.POINTER(HE_CKKS_RelinKey),
                ctypes.POINTER(HE_CKKS_SecretKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.c_int
        )

        self.CreateGaloisKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_Create,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.c_bool   #store_in_gpu
            ],
            restype=ctypes.POINTER(HE_CKKS_GaloisKey)
        )
        self.CreateGaloisKeyWithShifts = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_Create_With_Shifts,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(ctypes.c_int),   #shift_vec
                ctypes.c_size_t #shift_vec len
            ],
            restype=ctypes.POINTER(HE_CKKS_GaloisKey)
        )
        # I assume this means Galois Keys
        self.GenerateGaloisKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_KeyGenerator_GenerateGaloisKey,
            argtypes=[
                ctypes.POINTER(HE_CKKS_KeyGenerator),
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.POINTER(HE_CKKS_SecretKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.c_int
        )

        #For testing:
        self.generateSecretAndPublicKey = HEonGPUFunction(
            self.lib.generateSecretAndPublicKey,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=None
        )

        # HEonGPU does have serialization/load functions for it's keys,
        # but I must confirm that the functionality is the same
        # self.SerializeSecretKey = HEonGPUFunction(
        #     self.lib.SerializeSecretKey,
        #     argtypes=[],
        #     restype=ArrayResultByte
        # )

        # self.LoadSecretKey = HEonGPUFunction(
        #     self.lib.LoadSecretKey,
        #     argtypes=[ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong],
        #     restype=None
        # )

        self.SaveGaloisKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_Save, 
            [ctypes.POINTER(HE_CKKS_GaloisKey), ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_size_t)], 
            ctypes.c_int
        )
        self._DeleteGaloisKey = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_Delete, 
            [ctypes.POINTER(HE_CKKS_GaloisKey)], 
            None
        )
        self.StoreGaloisKeyInDevice = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_StoreInDevice,
            argtypes=[
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.c_void_p  # Corresponds to cudaStream_t, can be None
            ],
            restype=None
        )
        #change to be explicitly set in pinned memory
        self.StoreGaloisKeyInHost = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_GaloisKey_StoreInHost,
            argtypes=[
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.c_void_p  # Corresponds to cudaStream_t, can be None
            ],
            restype=None
        )

    
    def NewKeyGenerator(self):
        self.keygenerator_handle = self._NewKeyGenerator(self.context_handle)
        return self.keygenerator_handle
    
    def GenerateSecretKey(self):
        self.secretkey_handle = self.CreateSecretKey(self.context_handle)
        return self._GenerateSecretKey(self.keygenerator_handle, self.secretkey_handle, None)

    def GeneratePublicKey(self):
        self.publickey_handle = self.CreatePublicKey(self.context_handle)
        x = self._GeneratePublicKey(self.keygenerator_handle, self.publickey_handle, self.secretkey_handle, None)
        return x
    def DeleteGaloisKey(self, handle):
        return
        self._DeleteGaloisKey(handle)

    def GenerateRelinearizationKey(self):
        print("here")
        self.relinkey_handle = self.CreateRelinearizationKey(self.context_handle, False)
        return self._GenerateRelinearizationKey(self.keygenerator_handle, self.relinkey_handle, self.secretkey_handle, None)
    
    def GenerateEvaluationKeys(self):
        # self.galoiskey_handle = self.CreateGaloisKey(self.context_handle, False)
        # return self.GenerateGaloisKey(self.keygenerator_handle, self.galoiskey_handle, self.secretkey_handle, None)
        pass

    #encoder
    def setup_encoder(self):
        self._NewEncoder = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encoder_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context)],
            restype=ctypes.POINTER(HE_CKKS_Encoder)
        )

        self._Encode = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encoder_Encode_Double,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Encoder),    #encoder
                ctypes.POINTER(HE_CKKS_Plaintext),  #plaintext
                ctypes.POINTER(ctypes.c_double),    #data
                ctypes.c_size_t,                       #length
                ctypes.c_double,                    #scale
                ctypes.POINTER(C_ExecutionOptions)  
            ],
            restype=ctypes.c_int
        )
        self._Decode = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encoder_Decode_Double,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Encoder),   # encoder
                ctypes.POINTER(HE_CKKS_Plaintext), # pt
                ctypes.POINTER(ctypes.c_double),   # message_buffer (output)
                ctypes.c_size_t,                   # buffer_len
                ctypes.POINTER(C_ExecutionOptions)   # c_options
            ],
            restype=ctypes.c_int # Returns the number of elements copied
        )


    def NewEncoder(self):
        self.encoder_handle =  self._NewEncoder(self.context_handle)
    def Encode(self, to_encode, level, scale):
        scale = self.scale
        slot_count = self.poly_degree // 2 
        vector_size = len(to_encode)
        print(f"--- Python Encode Debug ---")
        print(f"  Slot Count Available: {slot_count}")
        print(f"  Requested Vector Size: {vector_size}")
        print(f"  Requested Scale: {scale}")
        pt = self.CreatePlaintext()
        status = self._Encode(self.encoder_handle, pt, to_encode, scale, None)
        if status != 0:
            raise RuntimeError(f"HEonGPU_CKKS_Encoder_Encode_Double failed with status {status}")
        return pt
  
    def EncodeSingle(self, scalar, level):
        # Use the existing Encode function by wrapping the scalar in a list.
        # The standard Encode function already handles creating and returning the
        # new plaintext handle.
        default_scale = self.scale
        return self.Encode([scalar], level, default_scale)

    def Decode(self, pt):
        num_slots = self.poly_degree // 2
        if not self.encoder_handle:
            raise RuntimeError("Encoder has not been initialized.")
        if not pt:
            raise ValueError("Input plaintext handle cannot be null.")
        buffer_len = num_slots
        message_buffer = (ctypes.c_double * buffer_len)()
        # print(message_buffer)
        elements_copied = self._Decode(
            self.encoder_handle,
            pt,
            message_buffer,
            buffer_len,
            None  # Assuming default execution options
        )

        if elements_copied < 0:
            raise RuntimeError(f"The C++ Decode operation failed with error code: {elements_copied}")

        # 4. Convert the ctypes array into a Python list and return it.
        # We slice it to the number of elements that were actually copied.
        return list(message_buffer[:elements_copied])








    #encryptor
    def setup_encryptor(self):
        self._NewEncryptor = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encryptor_Create_With_PublicKey,
            argtypes=[ctypes.POINTER(HE_CKKS_Context),
            ctypes.POINTER(HE_CKKS_PublicKey)],
            restype=ctypes.POINTER(HE_CKKS_Encryptor)
        )

        self._NewDecryptor = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Decryptor_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context),
            ctypes.POINTER(HE_CKKS_SecretKey)],
            restype=ctypes.POINTER(HE_CKKS_Decryptor)
        )

        #Originally argtypes were ctypes.c_int, which is incompatible with HEonGPU, hopefully this works
        self._Encrypt = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encryptor_Encrypt_New,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Encryptor),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._Decrypt = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Decryptor_Decrypt,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Decryptor),  # decryptor handle
                ctypes.POINTER(HE_CKKS_Plaintext),  # pt_out_c (output plaintext handle)
                ctypes.POINTER(HE_CKKS_Ciphertext), # ct_in_c (input ciphertext handle)
                ctypes.POINTER(C_ExecutionOptions)   # options
            ],
            restype=ctypes.c_int  # Returns an integer status code
        )

    
    def NewEncryptor(self):
        #assume 1 global public key for public key encryption:
        self.encryptor_handle = self._NewEncryptor(self.context_handle, self.publickey_handle)
    def NewDecryptor(self):
        #assume 1 global secret key for public key encryption:
        self.decryptor_handle = self._NewDecryptor(self.context_handle, self.secretkey_handle)
    def Encrypt(self, pt):
        new_ciphertext_handle = self._Encrypt(self.encryptor_handle, pt, None)
        if not new_ciphertext_handle:
            raise RuntimeError("HEonGPU_CKKS_Encryptor_Encrypt_New failed and returned a null ciphertext pointer.")
        return new_ciphertext_handle

    def Decrypt(self, ct):
        if not self.decryptor_handle:
            raise RuntimeError("Decryptor has not been initialized.")
        if not ct:
            raise ValueError("Input ciphertext handle cannot be null.")
        pt_out_handle = self.CreatePlaintext()
        # print("[DEBUG] Preparing to call self._Decrypt.")
        # print(f"    - Arg 'decryptor_handle': {self.decryptor_handle}")
        # print(f"    - Arg 'pt_out_handle' (output buffer): {pt_out_handle}")
        # print(f"    - Arg 'ct' (input ciphertext): {ct}")

        status = self._Decrypt(
            self.decryptor_handle,
            pt_out_handle,
            ct,
            None  # Assuming default execution options
        )
        if status != 0:
            raise RuntimeError(f"The C++ Decrypt operation failed with status code: {status}")
        return pt_out_handle







    def DeleteScheme(self):
        print("[DEBUG] In DeleteScheme", flush=True)
        self.HEonGPU_CKKS_SynchronizeDevice()
        self.shutdown()
        self.context_handle = None

    def DeleteBootstrappers(self):
        pass


    #evaluator:
    def setup_evaluator(self):
        self._NewEvaluator = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Create,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(HE_CKKS_Encoder)
            ],
            restype=ctypes.POINTER(HE_CKKS_ArithmeticOperator)
        )

        self.NewPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(HE_CKKS_Plaintext)
        )
        self.NewCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_Create,
            argtypes=[ctypes.POINTER(HE_CKKS_Context),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._ModDropCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_ModDrop_Ciphertext_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=None
        )

        #TODO: Determine the intention of this function, and map on correct HEonGPU 
        # self.AddRotationKey = HEonGPUFunction(
        #     self.lib.AddRotationKey,
        #     argtypes=[ctypes.c_int],
        #     restype=None
        # )

        self._Negate = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Negate,
            argtypes=[ctypes.POINTER(HE_CKKS_ArithmeticOperator),
            ctypes.POINTER(HE_CKKS_Ciphertext),
            ctypes.POINTER(HE_CKKS_Ciphertext),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )

        self._Rotate = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Rotate_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.c_int,    #amount
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=None
        )
        self._RotateNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Rotate,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.c_int,    #amount
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )   

        self._Rescale = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Rescale_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )

        # self._RescaleNew = HEonGPUFunction(
        #     self.lib.HEonGPU_CKKS_ArithmeticOperator_ModDrop_Ciphertext,
        #     argtypes=[
        #         ctypes.POINTER(HE_CKKS_ArithmeticOperator),
        #         ctypes.POINTER(HE_CKKS_Ciphertext),
        #         ctypes.POINTER(HE_CKKS_Ciphertext),
        #         ctypes.POINTER(C_ExecutionOptions)
        #         ],
        #     restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        # )

        self._SubPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._SubPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Plain,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )

        self._AddPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._AddPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Plain,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._MultiplyPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._MultiplyPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Plain,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        #ciphertext:
        self._AddCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._AddCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._SubCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._SubCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._MultiplyCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._MultiplyCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self._RelinearizeCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Relinearize_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(HE_CKKS_RelinKey),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._ModDropPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_ModDrop_Plaintext_Inplace,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=None  # The C++ function returns void
        )
        self._TestBootstrap = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_BootstrapTest,
            argtypes=[
                ctypes.POINTER(HE_CKKS_Context),
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_KeyGenerator),
                ctypes.POINTER(HE_CKKS_SecretKey),
                ctypes.POINTER(HE_CKKS_RelinKey),
                ctypes.POINTER(HE_CKKS_Encoder),
                ctypes.POINTER(HE_CKKS_Encryptor)
            ],
            restype=None 
        )

    def TestBootstrap(self):
        self._TestBootstrap(self.context_handle,
                            self.arithmeticoperator_handle,
                            self.keygenerator_handle,
                            self.secretkey_handle,
                            self.relinkey_handle,
                            self.encoder_handle,
                            self.encryptor_handle)
        return 0

    def ModDropCiphertextInplace(self, ct):
        if not self.arithmeticoperator_handle:
            raise RuntimeError("ArithmeticOperator has not been initialized.")
        if not ct:
            raise ValueError("Input ciphertext handle cannot be null.")

        self._ModDropCiphertext(
            self.arithmeticoperator_handle,
            ct,
            None # Assuming default execution options
        )
        return ct
    def ModDropPlaintext(self, ptxt):
        # This calls the in-place mod drop for plaintexts
        self._ModDropPlaintext(self.arithmeticoperator_handle, ptxt, None)
    def CreateCiphertext(self):
        return self.NewCiphertext(self.context_handle, None)
    def CreatePlaintext(self):
        return self.NewPlaintext(self.context_handle, None)
    def DeletePlaintext(self, pt):
        pass
        # return self.DeletePlaintext(pt)
    def NewEvaluator(self):
        print(self.context_handle)
        print(self.encoder_handle)
        self.arithmeticoperator_handle = self._NewEvaluator(self.context_handle, self.encoder_handle)
    def Negate(self, ct):
        #Not in place negation: must create empty ct for output first
        newct = self.NewCiphertext(self.context_handle, None)
        return self._Negate(self.arithmeticoperator_handle, ct, newct, None)
    def Rotate(self, ct, slots):
        self.HEonGPU_CKKS_SynchronizeDevice()
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # self.DeletePlaintext(newpt)
        # print(f"[DEBUG] In Rotate, old value is: {newval[:10]}")

        rotation_amount = slots
        if not hasattr(self, 'consolidated_galois_key_handle'):
            raise RuntimeError("Consolidated Galois key was not generated before calling Rotate.")

        if not hasattr(self, 'rotation_keys_cache'):
            self.rotation_keys_cache = []
        if not hasattr(self, 'normalized_steps'):
            self.normalized_steps = []   
        
        print(f"Trying to rotat by {rotation_amount}, supportd rotations are {self.normalized_steps}")
        if(rotation_amount not in self.normalized_steps and rotation_amount not in self.rotation_keys_cache):
            # Generate Key:
            # print(f"WARNING: On-the-fly key generation for rotation {rotation_amount}. This will be slow.")
            # try:
            #     self.GenerateLinearTransformRotationKey(rotation_amount)
            # except Exception as e:
            #     raise RuntimeError(f"On-the-fly key generation failed for rotation {rotation_amount}.") from e
            # Use power of 2 rotations
            powers = []
            power_of_2 = 1
            n = rotation_amount
            while n > 0:
                if n & 1:
                    powers.append(power_of_2)
                n >>= 1  # Right-shift n to check the next bit
                power_of_2 <<= 1  # Left-shift power_of_2 to double it

            for rot in powers:
                self.Rotate(ct, rot)
            
            return ct



        
        elif (rotation_amount in self.normalized_steps):  #key exists
            key_handle = self.consolidated_galois_key_handle
            # self.StoreGaloisKeyInDevice(key_handle, None)

            self.HEonGPU_CKKS_SynchronizeDevice()
            self._Rotate(self.arithmeticoperator_handle, ct, rotation_amount, key_handle, None)
            
            # self.StoreGaloisKeyInHost(key_handle, None)
            
            self.HEonGPU_CKKS_SynchronizeDevice()
            # newpt = self.Decrypt(ct)
            # newval =  self.Decode(newpt)
            # self.DeletePlaintext(newpt)
            # print(f"[DEBUG] In Rotate, new value is: {newval[:10]}")
            return ct
        elif (rotation_amount in self.rotation_keys_cache):
            specific_galois_key_handle = self.rotation_keys_cache[rotation_amount]
            # self.StoreGaloisKeyInDevice(specific_galois_key_handle, None)
            self._Rotate(self.arithmeticoperator_handle, ct, rotation_amount, specific_galois_key_handle, None)
            # self.StoreGaloisKeyInHost(specific_galois_key_handle, None)
            return ct

    def RotateNew(self, ct, slots):
        self.HEonGPU_CKKS_SynchronizeDevice()
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # self.DeletePlaintext(newpt)
        # print(f"[DEBUG] In RotateNew, old value is: {newval[:10]}")

        rotation_amount = slots

        if not hasattr(self, 'rotation_keys_cache'):
            self.rotation_keys_cache = []
        if not hasattr(self, 'normalized_steps'):
            self.normalized_steps = []
        
        print(f"Trying to rotat by {rotation_amount}, supportd rotations are {self.normalized_steps}")
        if(rotation_amount not in self.normalized_steps and rotation_amount not in self.rotation_keys_cache):
            # Generate Key:
            # print(f"WARNING: On-the-fly key generation for rotation {rotation_amount}. This will be slow.")
            # try:
            #     self.GenerateLinearTransformRotationKey(rotation_amount)
            #     self.HEonGPU_CKKS_SynchronizeDevice()
            # except Exception as e:
            #     raise RuntimeError(f"On-the-fly key generation failed for rotation {rotation_amount}.") from e
            # Use power of 2 rotations
            powers = []
            power_of_2 = 1
            n = rotation_amount
            while n > 0:
                if n & 1:
                    powers.append(power_of_2)
                n >>= 1  # Right-shift n to check the next bit
                power_of_2 <<= 1  # Left-shift power_of_2 to double it

            newct = self.RotateNew(ct, powers[0])
            powers.pop(0)
            for rot in powers:
                self.Rotate(newct, rot)
            self.HEonGPU_CKKS_SynchronizeDevice()
            return newct


        #Now, use key
        if (rotation_amount in self.normalized_steps):  #key exists
            key_handle = self.consolidated_galois_key_handle
            # self.StoreGaloisKeyInDevice(key_handle, None)

            newct_shell = self.NewCiphertext(self.context_handle, None)
            self.HEonGPU_CKKS_SynchronizeDevice()
            newct_result = self._RotateNew(self.arithmeticoperator_handle, ct, newct_shell, rotation_amount, key_handle, None)
            
            # self.StoreGaloisKeyInHost(key_handle, None)
            
            if not newct_result:
                raise RuntimeError(f"HEonGPU_CKKS_ArithmeticOperator_Rotate failed for rotation {rotation_amount} and returned a null pointer.")
            self.HEonGPU_CKKS_SynchronizeDevice()
            # newpt = self.Decrypt(newct_result)
            # newval =  self.Decode(newpt)
            # self.DeletePlaintext(newpt)
            # print(f"[DEBUG] In RotateNew1, new value is: {newval[:10]}")
            return newct_result
        elif (rotation_amount in self.rotation_keys_cache):
            specific_galois_key_handle = self.rotation_keys_cache[rotation_amount]
            newct_shell = self.NewCiphertext(self.context_handle, None)
            # self.StoreGaloisKeyInDevice(specific_galois_key_handle, None)
            self.HEonGPU_CKKS_SynchronizeDevice()
            newct = self._RotateNew(self.arithmeticoperator_handle, ct, newct_shell, rotation_amount, specific_galois_key_handle, None)
            # self.StoreGaloisKeyInHost(specific_galois_key_handle, None)
            self.HEonGPU_CKKS_SynchronizeDevice()
            # newpt = self.Decrypt(newct)
            # newval =  self.Decode(newpt)
            # self.DeletePlaintext(newpt)
            # print(f"[DEBUG] In RotateNew2, new value is: {newval[:10]}")
            return newct












    def Rescale(self, ct):
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # self.DeletePlaintext(newpt)
        # print(f"[DEBUG] In Rescale, old value is: {newval[:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        ct = ct.values if isinstance(ct, CipherTensor) else ct

        print(f"[DEBUG] In bindings.py Rescale, address of ct: {id(ct)}")
        print(f"[DEBUG] In bindings.py Rescale, object is: {object.__repr__(ct)}")

        self._Rescale(self.arithmeticoperator_handle, ct, None)
        if not ct:
            raise RuntimeError(f"The C++ Rescale_Inplace operation failed.")

        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # self.DeletePlaintext(newpt)
        # print(f"[DEBUG] In Rescale, new value is: {newval[:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        return ct
    def RescaleNew(self, ct):
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # self.DeletePlaintext(newpt)
        # print(f"[DEBUG] In RescaleNew, old value is: {newval[:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        cloned_ct = self.CloneCiphertext(ct)
        if not cloned_ct:
            raise RuntimeError("Failed to clone ciphertext in RescaleNew.")
        rescaled_clone = self.Rescale(cloned_ct)
        return rescaled_clone

    #For scalar operations, we must first encode scalar into a plaintext
    def SubScalar(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        self.SubPlaintext(ct, pt)
        self.DeletePlaintext(pt)

    def SubScalarNew(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        new_ct = self.SubPlaintextNew(ct, pt)
        self.DeletePlaintext(pt)
        return new_ct

    def AddScalar(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        ct_depth = self.GetCiphertextDepth(ct)
        pt = self.Encode(values, 0, self.scale)
        if ct_depth > 0:
            print(f"[DEBUG] Dropping plaintext modulus {ct_depth} times to match ciphertext.")
            for _ in range(ct_depth):
                self.ModDropPlaintext(pt)
        self.AddPlaintext(ct, pt)
        
        self.DeletePlaintext(pt)


    def AddScalarNew(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        new_ct = self.AddPlaintextNew(ct, pt)
        self.DeletePlaintext(pt)
        return new_ct

    def MulScalarInt(self, ctxt, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        self.MulPlaintext(ctxt, pt)
        self.DeletePlaintext(pt)

    def MulScalarIntNew(self, ctxt, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        new_ct = self.MulPlaintextNew(ctxt, pt)
        self.DeletePlaintext(pt)
        return new_ct

    def MulScalarFloat(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        self.MulPlaintext(ct, pt)
        self.DeletePlaintext(pt)

    def MulScalarFloatNew(self, ct, scalar):
        num_slots = self.poly_degree // 2
        values = [scalar] * num_slots
        pt = self.Encode(values, 0, self.scale)
        new_ct = self.MulPlaintextNew(ct, pt)
        self.DeletePlaintext(pt)
        return new_ct







    #these functions should be simple wrappers
    def AddPlaintext(self, ct, pt):
        print("\n--- [DEBUG] Entering AddPlaintext ---")
        ct_depth = self.GetCiphertextDepth(ct)
        pt_depth = self.GetPlaintextDepth(pt)
        print(f"ct_depth is {ct_depth} and pt_depth is {pt_depth}")
        if ct_depth > pt_depth:
            print(f"[DEBUG] Dropping plaintext modulus {ct_depth} times to match ciphertext.")
            for _ in range(ct_depth - pt_depth):
                self.ModDropPlaintext(pt)
        ct_level = self.GetCiphertextLevel(ct)
        pt_depth = self.GetPlaintextDepth(pt)
        print(f"    Ciphertext Level (remaining moduli): {ct_level}")
        print(f"    Plaintext Depth (consumed moduli): {pt_depth}")
        # oldpt = self.Decrypt(ct)
        # oldval =  self.Decode(oldpt)
        # print(f"[DEBUG] In AddPlaintext, old value is: {oldval[:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()

        print("--- [DEBUG] Calling backend _AddPlaintext ---")
        ret = self._AddPlaintext(self.arithmeticoperator_handle, ct, pt, None)
        
        self.HEonGPU_CKKS_SynchronizeDevice()
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # print(f"[DEBUG] In AddPlaintext, new value is: {newval[:10]}")

        return ret


    def AddPlaintextNew(self, ct, pt):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._AddPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct, None)
    def SubPlaintext(self, ct, pt):
        return self._SubPlaintext(self.arithmeticoperator_handle, ct, pt, None)
    def SubPlaintextNew(self, ct, pt):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._SubPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct, None)
    def MulPlaintext(self, ct, pt):
        ct_depth = self.GetCiphertextDepth(ct)
        pt_depth = self.GetPlaintextDepth(pt)
        print("Decoding in Multi")
        
        if ct_depth > pt_depth:
            print(f"[DEBUG] Dropping plaintext modulus {ct_depth} times to match ciphertext.")
            for _ in range(ct_depth - pt_depth):
                self.ModDropPlaintext(pt)
        print("About to call _MultiplyPlaintext")
        self._MultiplyPlaintext(self.arithmeticoperator_handle, ct, pt, None)
        return ct
    def MulPlainNew(self, ct, pt):
        newct_shell = self.NewCiphertext(self.context_handle, None)
        ct_depth = self.GetCiphertextDepth(ct)
        pt_depth = self.GetPlaintextDepth(pt)
        
        if ct_depth > pt_depth:
            print(f"[DEBUG] Dropping plaintext modulus {ct_depth} times to match ciphertext.")
            for _ in range(ct_depth - pt_depth):
                self.ModDropPlaintext(pt)
        print(newct_shell)
        newct_result = self._MultiplyPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct_shell, None)
        if not newct_result:
            raise RuntimeError("HEonGPU_CKKS_ArithmeticOperator_Multiply_Plain failed and returned a null pointer.")
        return newct_result
    
    def MulPlaintextNew(self, ct, pt):
        
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # print(f"[DEBUG], in MulPlaintextNew - {newval[:10]}")
        return self.MulPlainNew(ct, pt)
    
    def AddCiphertext(self, ct1, ct2):
        return self._AddCiphertext(self.arithmeticoperator_handle, ct1, ct2, None)
    def AddCiphertextNew(self, ct1, ct2):
        depth1 = self.GetCiphertextDepth(ct1)
        depth2 = self.GetCiphertextDepth(ct2)
        ct1_to_add = ct1
        ct2_to_add = ct2
        temp_ct = None

        try:
            if depth1 < depth2:
                print(f"[DEBUG] AddCiphertextNew: Mismatch. Cloning and dropping ct1 from depth {depth1} to {depth2}.")
                for _ in range(depth2 - depth1):
                    self.ModDropCiphertextInplace(ct1_to_add)

            elif depth2 < depth1:
                print(f"[DEBUG] AddCiphertextNew: Mismatch. Cloning and dropping ct2 from depth {depth2} to {depth1}.")
                for _ in range(depth1 - depth2):
                    self.ModDropCiphertextInplace(ct2_to_add)

            newct_shell = self.NewCiphertext(self.context_handle, None)


            # midpt = self.Decrypt(ct1_to_add)
            # midval =  self.Decode(midpt)
            # midpt2 = self.Decrypt(ct2_to_add)
            # midval2 =  self.Decode(midpt2)
            # print(f"[DEBUG] In AddCiphertextNew, adding {midval[:10]} with {midval2[:10]}")
            # print(f"[DEBUG] In AddCiphertextNew, ct1 has scale {self.GetCiphertextScale(ct1)}, ct2 has scale {self.GetCiphertextScale(ct2)}")
            self.HEonGPU_CKKS_SynchronizeDevice()
            newct_result = self._AddCiphertextNew(
                self.arithmeticoperator_handle,
                ct1_to_add,
                ct2_to_add,
                newct_shell,
                None
            )
            # endpt = self.Decrypt(newct_result)
            # endval =  self.Decode(endpt)
            # print(f"[DEBUG] In AddCiphertextNew, end is {endval[:10]}")
            self.HEonGPU_CKKS_SynchronizeDevice()
            if not newct_result:
                raise RuntimeError("HEonGPU_CKKS_ArithmeticOperator_Add failed and returned a null pointer.")

            return newct_result

        finally:
            if temp_ct:
                self.DeleteCiphertext(temp_ct)

    def SubCiphertext(self, ct1, ct2):
        return self._SubCiphertext(self.arithmeticoperator_handle, ct1, ct2, None)
    def SubCiphertextNew(self, ct1, ct2):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._SubCiphertextNew(self.arithmeticoperator_handle, ct1, ct2, newct, None)
    #combines multiplication and relin
    def MulRelinCiphertext(self, ct1, ct2):
        self._MultiplyCiphertext(self.arithmeticoperator_handle, ct1, ct2, None)
        self._RelinearizeCiphertext(self.arithmeticoperator_handle, ct1, self.relinkey_handle, None)
    #not currently implemented in HEonGPU wrapper
    def MulRelinCiphertextNew(self, ctxt0, ctxt1):
        ctxt_copy = self.CloneCiphertext(ctxt0)
        self.MulRelinCiphertext(ctxt_copy, ctxt1)
        return ctxt_copy


    #setup_poly_evaluator
    def setup_poly_evaluator(self):
        #No internal wrapped functions (for now)
        pass
    

    def NewPolynomialEvaluator(self):   
        #we're just going to use aerithmetic operator, so for this step all we do is setup an array
        self.polys = []

    def GenerateMonomial(self, coeffsPtr, coeffsLen):
        #Given an array of coefficients and the length of the array, save it to the array setup in NewPolynomialEvaluator, and return the index
        self.polys.append(coeffsPtr)
        return len(self.polys) - 1

    def GenerateChebyshev(self, coeffsPtr, coeffsLen):
        #Given an array of coefficients and the length of the array, save it to the array setup in NewPolynomialEvaluator, and return the index
        #for now, the internal representation will be identiacal
        return self.GenerateMonomial(coeffsPtr, coeffsLen)

    def EvaluatePolynomial(self, ctxt_in_handle, poly_id, out_scale=None):
        #we follow lattigo's approach here, for Monomial we simply use horner's method
        #for Chebyshev we will treat it as the same for now, but in the future we can optimize by using Clenshaw's Algorithm
        #Horner's method: y = ((...((c_d*x + c_{d-1})*x + ...)*x + c_0)
        #The list is assumed to be [c_0, c_1, ..., c_d], so we process it backwards
        print("[DEBUG] In EvaluatePolynomial")
        coeffs = self.polys[poly_id]
        degree = len(coeffs) - 1
        current_scale = self.GetCiphertextScale(ctxt_in_handle)
        current_level = self.GetCiphertextLevel(ctxt_in_handle)
        highest_coeff_ptxt = self.Encode([coeffs[-1]], level=current_level, scale=current_scale)
        result_ctxt_handle = self.Encrypt(highest_coeff_ptxt)
        self.DeletePlaintext(highest_coeff_ptxt)
        for i in range(degree - 1, -1, -1):
            self.MulRelinCiphertext(result_ctxt_handle, ctxt_in_handle)
            self.Rescale(result_ctxt_handle)
            next_coeff = coeffs[i]
            rescaled_scale = self.GetCiphertextScale(result_ctxt_handle)
            coeff_ptxt = self.Encode([next_coeff], level=self.GetCiphertextLevel(result_ctxt_handle), scale=rescaled_scale)
            self.AddPlaintext(result_ctxt_handle, coeff_ptxt)
            self.DeletePlaintext(coeff_ptxt)
        return result_ctxt_handle

    def GenerateMinimaxSignCoeffs(self, degrees, prec=64, logalpha=12, logerr=12, debug=1):
        #Use Lattigo's backend for this:
        from orion.backend.lattigo.bindings import LattigoFunction
        base_path = Path(__file__).parent.parent # Goes up from heongpu/bindings.py to orion/backend/
        lattigo_so_path = base_path / "lattigo" / "lattigo-linux.so"
        if not lattigo_so_path.exists():
            raise FileNotFoundError(f"Lattigo library not found at: {lattigo_so_path}")
        
        lattigo_lib = ctypes.CDLL(str(lattigo_so_path))


        self._GenerateMinimaxSignCoeffs = HEonGPUFunction(
            lattigo_lib.GenerateMinimaxSignCoeffs,
            argtypes=[
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, # degrees
                ctypes.c_int, # prec 
                ctypes.c_int, # logalpha
                ctypes.c_int, # logerr
                ctypes.c_int, # debug
            ],
            restype=ArrayResultDouble
        )
        res = self._GenerateMinimaxSignCoeffs(degrees, prec, logalpha, logerr, debug)
        print(res)
        return res


    #setup_lt_evaluator
    def setup_lt_evaluator(self):
        #currently set up at python level, so no backend binding necassary
        pass

    def NewLinearTransformEvaluator(self):
        self.linear_transforms = []
    
    def GenerateLinearTransform(self, diags_idxs, diags_data, level, bsgs_ratio=1.0, io_mode="none"):
        num_slots = self.poly_degree//2 
        #Un-flatten the diagonal data into a dictionary for easy access
        diagonals = {}
        offset = 0
        for idx in diags_idxs:
            diagonals[idx] = diags_data[offset : offset + num_slots]
            offset += num_slots
        transform_plan = {
            "diagonals": diagonals,
            "level": level
            #more advanced implementation could store bsgs_ratio and io_mode (but for now we assume default)
        }
        self.linear_transforms.append(transform_plan)
        return len(self.linear_transforms) - 1
    


    def EvaluateLinearTransform(self, transform_id, ctxt_in_handle):
        # newpt = self.Decrypt(ctxt_in_handle)
        # newval =  self.Decode(newpt)
        # print(f"[EVALUATELINEARTRANSFORM BEFORE] - {newval[0:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        plan = self.linear_transforms[transform_id]
        diagonals = plan['diagonals']
        
        initial_scale = self.scale
        self.Rescale(ctxt_in_handle)
        initial_level = self.GetCiphertextLevel(ctxt_in_handle)
        num_slots = self.GetCiphertextSlots(ctxt_in_handle)
        zero_ptxt = self.Encode([0.0] * num_slots, level=initial_level, scale=initial_scale)
        accumulator_ctxt = self.Encrypt(zero_ptxt)
        self.DeletePlaintext(zero_ptxt) # This plaintext is no longer needed.
        for diag_idx, diag_coeffs in diagonals.items():
            print(f"[EVALUATELINEARTRANSFORM MIDDLE] Looping, diag_idx = {diag_idx}")
            print(f"[EVALUATELINEARTRANSFORM MIDDLE] ctxt_in_handle depth = {self.GetCiphertextDepth(ctxt_in_handle)}")
            self.HEonGPU_CKKS_SynchronizeDevice()
            rotated_ctxt = self.RotateNew(ctxt_in_handle, diag_idx)
            self.HEonGPU_CKKS_SynchronizeDevice()
            diag_ptxt = self.Encode(diag_coeffs, level=self.GetCiphertextLevel(rotated_ctxt), scale=initial_scale)
            
            if not rotated_ctxt or not diag_ptxt:
                raise RuntimeError("Failed to create rotated ciphertext or diagonal plaintext.")
            
            print(f"[EVALUATELINEARTRANSFORM MIDDLE rotated_ctxt scale - {self.GetCiphertextScale(rotated_ctxt)} diag_ptxt scale - {self.GetPlaintextScale(diag_ptxt)}")
            # newpt = self.Decrypt(rotated_ctxt)
            # newval =  self.Decode(newpt)
            # print(f"[EVALUATELINEARTRANSFORM MIDDLE Multiplying diag_coeffs - {diag_coeffs[0:10]} by rotated_ctxt - {newval[0:10]}")
            self.HEonGPU_CKKS_SynchronizeDevice()
            self.MulPlaintext(rotated_ctxt, diag_ptxt)
            term_ctxt = rotated_ctxt
            self.HEonGPU_CKKS_SynchronizeDevice()
            self.Rescale(term_ctxt)
            print(f"[EVALUATELINEARTRANSFORM MIDDLE term_ctxt scale - {self.GetCiphertextScale(term_ctxt)}")
            # term_ctxt = self.SetCiphertextScale(term_ctxt, self.GetCiphertextScale(accumulator_ctxt))
            if not term_ctxt:
                raise RuntimeError("Failed to multiply rotated ciphertext and diagonal plaintext.")

            self.HEonGPU_CKKS_SynchronizeDevice()
            # midpt2 = self.Decrypt(accumulator_ctxt)
            # midval2 =  self.Decode(midpt2)
            # midpt3 = self.Decrypt(term_ctxt)
            # midval3 =  self.Decode(midpt3)
            # print(f"[EVALUATELINEARTRANSFORM MIDDLE accumulator_ctxt - {midval2[:10]}")
            # print(f"[EVALUATELINEARTRANSFORM MIDDLE term_ctxt - {midval3[:10]}")
            while(self.GetCiphertextDepth(accumulator_ctxt)<self.GetCiphertextDepth(term_ctxt)):
                self.ModDropCiphertextInplace(accumulator_ctxt)
            self.AddCiphertext(accumulator_ctxt, term_ctxt)
            # midpt = self.Decrypt(new_accumulator_ctxt)
            # midval =  self.Decode(midpt)
            
            # print(f"[EVALUATELINEARTRANSFORM MIDDLE accumulator_ctxt - {midval[:10]}")
            self.HEonGPU_CKKS_SynchronizeDevice()
            if not accumulator_ctxt:
                raise RuntimeError("Failed to add term to accumulator.")

            self.DeletePlaintext(diag_ptxt)
            self.DeleteCiphertext(term_ctxt)
            if diag_idx != 0:
                self.DeleteCiphertext(rotated_ctxt)


        # newpt = self.Decrypt(accumulator_ctxt)
        # newval =  self.Decode(newpt)
        # print(f"[EVALUATELINEARTRANSFORM AFTER] - {newval[0:10]}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        return accumulator_ctxt
        









    def DeleteLinearTransform(self, transform_id):
        return
        if 0 <= transform_id < len(self.linear_transforms):
            self.linear_transforms[transform_id] = None
    

    def GetLinearTransformRotationKeys(self, transform_id):
        #inspects a compiled transform and returns the list of required rotation indices
        self.HEonGPU_CKKS_SynchronizeDevice()
        plan = self.linear_transforms[transform_id]
        #the diagonal indices are the rotation indices needed
        #rotation by 0 is a conjugation, requires specific galois key
        #Lattigo convention represent this with the Galois element for N+1
        #but we use a simple mapping: rotation_step=0 -> galois_elt=0 (placeholder)
        # required_rotations = list(plan['diagonals'].keys())
        required_rotations = [0]
        #also generate all powers of 2
        i = 1
        while i <= self.poly_degree // 2:
            if i not in required_rotations:
                required_rotations.append(i)
            i *= 2
        
        print(f"[DEBUG] Required Rotations: {required_rotations}")
        if 0 in required_rotations:
            #for conjugation
            pass
        return required_rotations

    def GenerateLinearTransformRotationKey(self, rotation_step):
        if rotation_step in self.rotation_keys_cache:
            return
        num_slots = self.poly_degree // 2
        normalized_step = rotation_step % num_slots
        print(f"INFO: Generating Galois key for rotation step {rotation_step} and storing on HOST.")

        galois_key_handle = self.CreateGaloisKeyWithShifts(self.context_handle, [normalized_step])
        if not galois_key_handle:
            raise RuntimeError(f"Failed to create GaloisKey object for step {rotation_step}")
        status = self.GenerateGaloisKey(
            self.keygenerator_handle,
            galois_key_handle,
            self.secretkey_handle,
            None
        )
        if status != 0:
            self.DeleteGaloisKey(galois_key_handle)
            raise RuntimeError(f"HEonGPU_CKKS_KeyGenerator_GenerateGaloisKey failed for step {rotation_step} with status {status}")
        # self.StoreGaloisKeyInHost(galois_key_handle, None) # Using null stream
        self.rotation_keys_cache[rotation_step] = galois_key_handle

    def GenerateConsolidatedRotationKeys(self, rotation_steps: list):
        #given list, split into consolidated keys of 10 rotations each
        # not done
        if hasattr(self, 'consolidated_galois_key_handle'):
            return
        num_slots = self.poly_degree // 2
        normalized_steps = [step % num_slots for step in rotation_steps]
        self.normalized_steps = normalized_steps
        print(f"INFO: Generating a single consolidated Galois key for {len(normalized_steps)} rotations.")
        galois_key_handle = self.CreateGaloisKeyWithShifts(self.context_handle, normalized_steps)
        if not galois_key_handle:
            raise RuntimeError(f"Failed to create consolidated GaloisKey object.")

        status = self.GenerateGaloisKey(
            self.keygenerator_handle,
            galois_key_handle,
            self.secretkey_handle,
            None
        )
        if status != 0:
            self.DeleteGaloisKey(galois_key_handle)
            raise RuntimeError(f"HEonGPU_CKKS_KeyGenerator_GenerateGaloisKey failed for consolidated key with status {status}")

        # self.StoreGaloisKeyInHost(galois_key_handle, None)
        self.consolidated_galois_key_handle = galois_key_handle



    def GenerateAndSerializeRotationKey(self, rotation_step):
        #Generates a specific Galois key and returns its serialized byte representation
        galois_key_handle = self.CreateGaloisKeyWithShifts(self.context_handle, [rotation_step])
        status = self.GenerateGaloisKey(
            self.keygenerator_handle, 
            galois_key_handle, 
            self.secretkey_handle, 
            None
        )
        serialized_key_bytes, _ = self.SaveGaloisKey(galois_key_handle)
        self.DeleteGaloisKey(galois_key_handle)
        return serialized_key_bytes

    def LoadRotationKey(self, serialized_key, rotation_step):
        #Deserializes a Galois key and caches it for later use by the evaluator
        galois_key_handle = self.LoadGaloisKey(serialized_key, [rotation_step])
        self.rotation_keys_cache[rotation_step] = galois_key_handle

    def SerializeDiagonal(self, transform_id, diag_idx):
        #Encodes and then serializes a specific diagonal from a transform plan
        plan = self.linear_transforms[transform_id]
        diag_coeffs = plan['diagonals'].get(diag_idx)
        scale = self.params.get_default_scale()
        level = plan['level']
        plaintext_handle = self.Encode(diag_coeffs, level=level, scale=scale)
        serialized_diag, _ = self.SavePlaintext(plaintext_handle)
        self.DeletePlaintext(plaintext_handle)
        return serialized_diag

    def LoadPlaintextDiagonal(self, serialized_diag, transform_id, diag_idx):
        #might be required later if implemented in c backend, but not now
        pass

    def RemovePlaintextDiagonals(self, transform_id):
        if 0 <= transform_id < len(self.linear_transforms):
            plan = self.linear_transforms[transform_id]
            if plan and 'diagonals' in plan:
                plan['diagonals'].clear()
    def RemoveRotationKeys(self):
        if hasattr(self, 'rotation_keys_cache'):
            for key_handle in self.rotation_keys_cache.values():
                if key_handle:
                    self.DeleteGaloisKey(key_handle)
            self.rotation_keys_cache.clear()

    def setup_bootstrapper(self):
        self._GenerateBootstrappingParams = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_GenerateBootstrappingParams,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.c_double,
                ctypes.POINTER(C_BootstrappingConfig)
            ],
            restype=ctypes.c_int 
        )
        self._GetBootstrappingKeyIndices = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_GetBootstrappingKeyIndices,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.c_uint64)
            ],
            restype=ctypes.c_int 
        )
        self._RegularBootstrapping = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_RegularBootstrapping,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.POINTER(HE_CKKS_RelinKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
    
    #should generally work (might overshoot)
    BOOTSTRAP_PRESET_CONFIG = {
        # Target Depth: { 'taylor_number', 'CtoS_piece', 'StoC_piece', 'less_key_mode' }
        1: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        2: { 
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        3: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        4: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        5: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        6: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        7: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        8: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        9: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        },
        10: {
            'taylor_number': 11, 
            'CtoS_piece': 3,
            'StoC_piece': 3,
            'less_key_mode': True
        }
    }
    def NewBootstrapper(self, logPs, num_slots):
        #we dont care about the exaxt logPs, or num_slots, just the length. HEonGPU will create the logPs from the parameters given:
        print(logPs)
        length = len(logPs)
        config_params = self.BOOTSTRAP_PRESET_CONFIG[length]
        print("[DEBUG] In NewBootstrapper:")
        print(f"    - logPs length: {length}")
        print(f"    - Selected config_params: {config_params}")
        print(f"    - Scale being passed: {self.scale}")

        boot_config = C_BootstrappingConfig(
            CtoS_piece=config_params['CtoS_piece'],
            StoC_piece=config_params['StoC_piece'],
            taylor_number=config_params['taylor_number'],
            less_key_mode=config_params['less_key_mode']
        )

        print("\n[DEBUG] Preparing to call _GenerateBootstrappingParams with the following arguments:")
        print(f"    - Arg 'arithmeticoperator_handle': {self.arithmeticoperator_handle}")
        print(f"    - Arg 'scale': {self.scale}")
        print(f"    - Arg 'boot_config':")
        print(f"        - CtoS_piece: {boot_config.CtoS_piece}")
        print(f"        - StoC_piece: {boot_config.StoC_piece}")
        print(f"        - taylor_number: {boot_config.taylor_number}")
        print(f"        - less_key_mode: {boot_config.less_key_mode}\n")
        status = self._GenerateBootstrappingParams(
            self.arithmeticoperator_handle,
            ctypes.c_double(self.scale),
            ctypes.byref(boot_config)
        )
        print("[DEBUG]   - Output of _GenerateBootstrappingParams:")
        print(f"[DEBUG]     - Status: {status}")
        self.HEonGPU_CKKS_SynchronizeDevice()
        indices_ptr = ctypes.POINTER(ctypes.c_int)()
        count = ctypes.c_size_t()
        
        status = self._GetBootstrappingKeyIndices(
            self.arithmeticoperator_handle,
            ctypes.byref(indices_ptr),
            ctypes.byref(count)
        )
        print("[DEBUG]   - Output of _GetBootstrappingKeyIndices:")
        print(f"[DEBUG]     - Status: {status}")
        print(f"[DEBUG]     - Count of indices: {count.value}")

            
        indices_list = [indices_ptr[i] for i in range(count.value)]
        print(f"[DEBUG]     -  indices: {indices_list}")

        self.bootstrap_galois_key_handle = self.CreateGaloisKeyWithShifts(
            self.context_handle,
            indices_list
        )
        print("[DEBUG]   - Output of CreateGaloisKeyWithShifts:")
        print(f"[DEBUG]     - Galois Key Handle: {self.bootstrap_galois_key_handle}")
        #Need to generate the keys here *****
        status = self.GenerateGaloisKey(
            self.keygenerator_handle,
            self.bootstrap_galois_key_handle,
            self.secretkey_handle,
            None
        )

    def Bootstrap(self, ct, num_slots):
        
        print("[DEBUG] Bootsrapping!")
        print("[DEBUG] Entering Python binding for Bootstrap.")
        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # print(f"[DEBUG], before Bootstrap - {newval[:10]}")
        # self.NewBootstrapper([60, 60], -1)  #delete later, but good for testing
        #Try to replicate the bootstrapping example from HEonGPU, key is to look at bindings and params/setup to ensure everything works (liekly doest)
        required_level = 1
        total_levels = self.q_size

        while True:
            current_depth = self.GetCiphertextDepth(ct)
            current_level = total_levels - current_depth

            if current_level <= required_level:
                print(f"[DEBUG]   - Ciphertext is at required level {current_level}. Proceeding.")
                break

            print(f"[DEBUG]   - Ciphertext level is {current_level}. Dropping to {current_level - 1} via ModDropCiphertextInplace.")
            

            self.ModDropCiphertextInplace(ct)

        ct_depth = self.GetCiphertextDepth(ct)
        ct_level = total_levels - ct_depth
        ct_scale = self.GetCiphertextScale(ct)

        print(f"[DEBUG]   - Final Input ciphertext depth from C++: {ct_depth}")
        print(f"[DEBUG]   - Final Input ciphertext remaining levels: {ct_level}")
        print(f"[DEBUG]   - Final Input ciphertext scale from C++: {ct_scale}")
        
        


        print("[DEBUG] Before _RegularBootstrapping.")
        bootstrapped_ct = self._RegularBootstrapping(
            self.arithmeticoperator_handle,
            ct,
            self.bootstrap_galois_key_handle,
            self.relinkey_handle,
            None 
        )

        if not bootstrapped_ct:
            raise RuntimeError("The C++ RegularBootstrapping operation failed and returned a null pointer.")

        # newpt = self.Decrypt(ct)
        # newval =  self.Decode(newpt)
        # print(f"[DEBUG], before Bootstrap - {newval[:10]}")
        print("[DEBUG] Finished Bootstrapping!")
        return bootstrapped_ct

    






    

