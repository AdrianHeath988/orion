import os 
import ctypes
import platform

import torch
import numpy as np
import hashlib

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

class ArrayResultInt(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_int)), 
                ("Length", ctypes.c_size_t)]

class ArrayResultFloat(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_float)), 
                ("Length", ctypes.c_size_t)]

class ArrayResultDouble(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_double)), 
                ("Length", ctypes.c_size_t)]

class ArrayResultUInt64(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_uint64)), # c_uint64 is an alias for c_ulonglong
                ("Length", ctypes.c_size_t)]

class ArrayResultByte(ctypes.Structure):
    _fields_ = [("Data", ctypes.POINTER(ctypes.c_ubyte)), # (unsigned char *)
                ("Length", ctypes.c_size_t)]


class HEonGPUFunction:
    """Helper to wrap ctypes functions with argument and return types."""
    def __init__(self, func, argtypes, restype):
        self.func = func
        self.func.argtypes = argtypes 
        self.func.restype = restype

    def __call__(self, *args):
        """
        Handles calling the C function with Python arguments and returning a Python result.
        """
        py_args_list = list(args)
        c_args_list = []
        
        # Use a while loop to manually control indices for both arg lists
        py_arg_idx = 0
        c_arg_type_idx = 0
        
        if self.func.argtypes:
            while py_arg_idx < len(py_args_list):
                current_py_arg = py_args_list[py_arg_idx]
                current_c_type = self.func.argtypes[c_arg_type_idx]
                if isinstance(current_py_arg, list) and isinstance(current_c_type, type(ctypes.POINTER(ctypes.c_void_p))):

                    len_c_type = self.func.argtypes[c_arg_type_idx + 1]
                    if len_c_type is not ctypes.c_size_t:
                        raise TypeError(f"C function signature is incorrect: Expected c_size_t after pointer for list argument, but got {len_c_type}")
                    c_arg_tuple = self.convert_to_ctypes(current_py_arg, current_c_type)
                    c_args_list.extend(c_arg_tuple)
                    py_arg_idx += 1
                    c_arg_type_idx += 2
                else:
                    c_arg = self.convert_to_ctypes(current_py_arg, current_c_type)
                    c_args_list.append(c_arg)
                    
                    py_arg_idx += 1
                    c_arg_type_idx += 1
        
        c_result = self.func(*c_args_list)
        py_result = self.convert_from_ctypes(c_result, self.func.restype)
        
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
            elif typ == ctypes.POINTER(ctypes.c_uint64):
                return ((ctypes.c_uint64 * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ulong):
                return ((ctypes.c_ulong * len(arg))(*arg), len(arg))
            elif typ == ctypes.POINTER(ctypes.c_ubyte):
                return ((ctypes.c_ubyte * len(arg))(*arg), len(arg))
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
            
    def convert_from_ctypes(self, res, restype):
        if (hasattr(restype, '_type_') and 
        isinstance(restype._type_, type) and 
        issubclass(restype._type_, ctypes.Structure)):
            if not res:
                return None
            if issubclass(restype, ArrayResultFloat):
                return [float(res.contents.Data[i]) for i in range(res.contents.Length)]
            elif issubclass(restype, (ArrayResultInt, ArrayResultUInt64)):
                return [int(res.contents.Data[i]) for i in range(res.contents.Length)]
            elif issubclass(restype, ArrayResultDouble):
                return [float(res.contents.Data[i]) for i in range(res.contents.Length)]
            elif issubclass(restype, ArrayResultByte):
                buffer = ctypes.cast(res.contents.Data, ctypes.POINTER(ctypes.c_ubyte * res.contents.Length)).contents
                array = np.frombuffer(buffer, dtype=np.uint8)
                return array, res.contents.Data 
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

    #setup scheme
    def setup_scheme(self, orion_params):
        """
        Initializes and configures the HEonGPU scheme by creating and setting up
        a CKKS context object.

        This function binds to the C API functions defined in context_c_api.h.
        """
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
        keyswitch_method = 1 
        sec_level = 128
        
        # Lattigo parameters mapped to HEonGPU
        poly_degree = 1 << orion_params.get_logn() # logn -> n
        poly_degree = 8192
        print(poly_degree)
        logq = orion_params.get_logq()             # Bit-lengths of Q primes
        logp = orion_params.get_logp()             # Bit-lengths of P primes
        # Unable to set scale directly in heongpu context, it's set in the encoder instead by user input
        scale = 1 << orion_params.get_logscale()   # logscale -> scale
        self.scale  = scale
        
        context_handle = self.HEonGPU_CKKS_Context_Create(keyswitch_method, sec_level)
        if not context_handle:
            raise RuntimeError("Failed to create HEonGPU CKKS Context.")

        self.HEonGPU_CKKS_Context_SetPolyModulusDegree(context_handle, poly_degree)
        result_modulus = self.HEonGPU_CKKS_Context_SetCoeffModulusBitSizes(context_handle, logq, logp[0:1])
        if result_modulus != 0:
            self.HEonGPU_CKKS_Context_Delete(context_handle)
            raise RuntimeError("Failed to set HEonGPU coefficient modulus bit-sizes.")

        print("before generation")
        result = self.HEonGPU_CKKS_Context_Generate(context_handle)
        if result != 0:
            self.HEonGPU_CKKS_Context_Delete(context_handle)
            raise RuntimeError("Failed to generate HEonGPU context parameters.")
        self.HEonGPU_PrintParameters(context_handle)

        print("HEonGPU CKKS Context successfully created and configured.")

        # context handle is required for all subsequent operations
        return context_handle
    #tensor binds
    def setup_tensor_binds(self):
        self.DeletePlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Plaintext_Delete,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
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

        #Not A thing
        # self.SetCiphertextScale = HEonGPUFunction(
        #     self.lib.HEonGPU_CKKS_Ciphertext_SetScale,
        #     argtypes=[
        #         ctypes.POINTER(HE_CKKS_Ciphertext),
        #         ctypes.c_double,
        #     ],
        #     restype=ctypes.c_double
        # )

        # "Level" corresponds to the number of remaining prime moduli in the chain.
        # However, plaintext in HEonGPU stores depth (number of prime moduli consumed).
        # Need to think on how to resolvbe this.
        # self.GetPlaintextLevel = HEonGPUFunction(
        #     self.lib.HEonGPU_CKKS_Plaintext_GetDepth,
        #     argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
        #     restype=ctypes.c_int
        # )

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


        self.GetCiphertextDegree = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Ciphertext_GetCiphertextSize,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.c_int
        )

        # Not a thing currently (check context_c_api.cu) to implement
        # self.GetModuliChain = HEonGPUFunction(
        #     self.lib.HEonGPU_CKKS_Context_GetCoeffModulus,
        #     argtypes=[
        #         ctypes.POINTER(HE_CKKS_Context),
        #         ctypes.POINTER(C_Modulus64),
        #         ctypes.c_size_t
        #     ],
        #     restype=ctypes.c_size_t
        # )
    
    def GetPlaintextSlots(self, plaintext_handle):
        if not plaintext_handle:
            raise ValueError("Invalid plaintext handle provided.")
        plain_size = self.GetPlaintextSize(plaintext_handle)
        return plain_size // 2
    
    def GetCiphertextSlots(self, ciphertext_handle):
        if not ciphertext_handle:
            raise ValueError("Invalid ciphertext handle provided.")
        ring_size = self.GetCiphertextSize(ciphertext_handle)
        return ring_size // 2

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
        # I assume this means Galois Key
        self.GenerateEvaluationKeys = HEonGPUFunction(
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

    def NewKeyGenerator(self):
        self.keygenerator = self._NewKeyGenerator(self.context_handle)
        return self.keygenerator
    
    def GenerateSecretKey(self):
        self.secretkey = self.CreateSecretKey(self.context_handle)
        return self._GenerateSecretKey(self.keygenerator, self.secretkey, None)

    def GeneratePublicKey(self):
        # self.publickey = self.CreatePublicKey(self.context_handle)
        # print("a")
        # print(self.keygenerator)
        # print(self.publickey)
        # print(self.secretkey)
        # x = self._GeneratePublicKey(self.keygenerator, self.publickey, self.secretkey, None)
        # print("b")
        # return x
        print("a")
        self.generateSecretAndPublicKey(self.context_handle, None)
        print("b")

    def GenerateRelinearizationKey(self):
        print("here")
        self.relinkey = self.CreateRelinearizationKey(self.context_handle)
        return self._GenerateRelinearizationKey(self.keygenerator, self.relinkey, self.secretkey, None)
    
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
                ctypes.c_int,                       #length
                ctypes.c_double,                    #scale
                ctypes.POINTER(C_ExecutionOptions)  
            ],
            restype=ctypes.c_int
        )
        #TODO: Determine what the argtypes of decode mean, and add wrapper function
        self.Decode = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_Encoder_Decode_Double,
            argtypes=[ctypes.c_int],
            restype=ArrayResultFloat,
        )

    def NewEncoder(self):
        self.encoder_handle =  self._NewEncoder(self.context_handle)
    def Encode(self, to_encode, level, scale):
        pt = self.NewPlaintext(self.context_handle, None)
        return self._Encode(self.encoder_handle, pt, to_encode, scale, None)
  
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
            restype=None
        )

        #Originally argtypes were ctypes.c_int, which is incompatible with HEonGPU, hopefully this works
        self.Encrypt = HEonGPUFunction(
            self.lib.Encrypt,
            argtypes=[ctypes.POINTER(HE_CKKS_Plaintext)],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
        self.Decrypt = HEonGPUFunction(
            self.lib.Decrypt,
            argtypes=[ctypes.POINTER(HE_CKKS_Ciphertext)],
            restype=ctypes.POINTER(HE_CKKS_Plaintext)
        )
    
    def NewEncryptor(self):
        #assume 1 global public key for public key encryption:
        self.encryptor_handle = self._NewEncryptor(self.context_handle, self.publickey_handle)
    def NewDecryptor(self):
        #assume 1 global secret key for public key encryption:
        self.decryptor_handle = self._NewDecryptor(self.context_handle, self.secretkey_handle)
    

    #evaluator:
    def setup_evaluator():
        self._NewEvaluator = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Create,
            argtypes=[ctypes.POINTER(ctypes.HE_CKKS_Context),
            ctypes.POINTER(ctypes.HE_CKKS_Encoder)],
            restype=ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator)
        )

        self.NewPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Create,
            argtypes=[ctypes.POINTER(ctypes.HE_CKKS_Context),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Plaintext)
        )
        self.NewCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Create,
            argtypes=[ctypes.POINTER(ctypes.HE_CKKS_Context),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )

        #TODO: Determine the intention of this function, and map on correct HEonGPU 
        self.AddRotationKey = LattigoFunction(
            self.lib.AddRotationKey,
            argtypes=[ctypes.c_int],
            restype=None
        )

        self._Negate = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Negate,
            argtypes=[ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
            ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
            ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
            ctypes.POINTER(C_ExecutionOptions)],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )

        self._Rotate = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Rotate_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.c_int,    #amount
                ctypes.POINTER(ctypes.HE_CKKS_GaloisKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=None
        )
        self._RotateNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Rotate,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.c_int,    #amount
                ctypes.POINTER(ctypes.HE_CKKS_GaloisKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )   

        self._Rescale = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_ModDrop_Ciphertext_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )

        self._RescaleNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_ModDrop_Ciphertext,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )

        self._SubPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._SubPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Plain,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )

        self._AddPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._AddPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Plain,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )
        self._MultiplyPlaintext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Plain_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._MultiplyPlaintextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Plain,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(ctypes.HE_CKKS_Plaintext),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )
        #ciphertext:
        self._AddCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._AddCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Add,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )
        self._SubCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._SubCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Sub,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )
        self._MultiplyCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        self._MultiplyCiphertextNew = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Multiply,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #out
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=ctypes.POINTER(ctypes.HE_CKKS_Ciphertext)
        )
        self._RelinearizeCiphertext = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_Relinearize_Inplace,
            argtypes=[
                ctypes.POINTER(ctypes.HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.HE_CKKS_Ciphertext),  #in/out
                ctypes.POINTER(ctypes.HE_CKKS_RelinKey),
                ctypes.POINTER(C_ExecutionOptions)
                ],
            restype=None
        )
        
        
    
    def NewEvaluator(self):
        self.arithmeticoperator_handle = self._NewEvaluator(self.context_handle, self.encoder_handle)
    def Negate(self, ct):
        #Not in place negation: must create empty ct for output first
        newct = self.NewCiphertext(self.context_handle, None)
        return self._Negate(self.arithmeticoperator_handle, ct, newct, None)
    def Rotate(self, ct, slots):
        return self._Rotate(self.arithmeticoperator_handle, ct, slots, self.galois_handle, None)
    def RotateNew(self, ct, slots):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._RotateNew(self.arithmeticoperator_handle, ct, newct, slots, self.galois_handle, None)
    def Rescale(self, ct):
        return self._Rescale(self.arithmeticoperator_handle, ct, None)
    def RescaleNew(self, ct):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._Rescale(self.arithmeticoperator_handle, ct, newct, None)

    #For scalar operations, we must first encode scalar into a plaintext
    def SubScalar(self, ct, scalar):
        #Must create plaintext then do sub

    def SubScalarNew(self, ct, scalar):
        
    def AddScalar():
    
    def AddScalarNew():

    def MulScalarInt():

    def MulScalarIntNew():
    
    def MulScalarFloat():

    def MulScalarFloatNew():
    
    #these functions should be simple wrappers
    def AddPlaintext(self, ct, pt):
        return self._AddPlaintext(self.arithmeticoperator_handle, ct, pt, None)
    def AddPlaintextNew(self, ct, pt):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._AddPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct, None)
    def SubPlaintext(self, ct, pt):
        return self._SubPlaintext(self.arithmeticoperator_handle, ct, pt, None)
    def SubPlaintextNew(self, ct, pt):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._SubPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct, None)
    def MulPlaintext(self, ct, pt):
        return self._MultiplyPlaintext(self.arithmeticoperator_handle, ct, pt, None)
    def MulPlaintextNew(self, ct, pt):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._MultiplyPlaintextNew(self.arithmeticoperator_handle, ct, pt, newct, None)
    def AddCiphertext(self, ct1, ct2):
        return self._AddCiphertext(self.arithmeticoperator_handle, ct1, ct2, None)
    def AddCiphertextNew(self, ct1, ct2):
        newct = self.NewCiphertext(self.context_handle, None)
        return self._AddCiphertextNew(self.arithmeticoperator_handle, ct1, ct2, newct, None)
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
    #def MulRelinCiphertextNew():

    #setup_poly_evaluator
    def setup_poly_evaluator(self):
        #No internal wrapped functions (for now)
    

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
        coeffs = self.polys[poly_id]
        degree = len(coeffs) - 1
        current_scale = self.GetCiphertextScale(ctxt_in_handle)
        current_level = self.GetCiphertextLevel(ctxt_in_handle)
        highest_coeff_ptxt = self.Encode([coeffs[-1]], level=current_level, scale=current_scale)
        result_ctxt_handle = self.Encrypt(highest_coeff_ptxt)
        self.DeletePlaintext(highest_coeff_ptxt)
        for i in range(degree - 1, -1, -1):
            self.MulRelin_Inplace(result_ctxt_handle, ctxt_in_handle)
            self.Rescale_Inplace(result_ctxt_handle)
            next_coeff = coeffs[i]
            rescaled_scale = self.GetCiphertextScale(result_ctxt_handle)
            coeff_ptxt = self.Encode([next_coeff], level=self.GetCiphertextLevel(result_ctxt_handle), scale=rescaled_scale)
            self.AddPlain_Inplace(result_ctxt_handle, coeff_ptxt)
            self.DeletePlaintext(coeff_ptxt)
        return result_ctxt_handle

    def GenerateMinimaxSignCoeffs(self, degrees, prec=64, logalpha=12, logerr=12, debug=False):-
        #Create a cache if it doesn't exist
        if not hasattr(self, 'minimax_sign_coeffs_cache'):
            self.minimax_sign_coeffs_cache = {}

        key_params = (tuple(degrees), prec, logalpha, logerr)
        param_hash = hashlib.sha256(str(key_params).encode()).hexdigest()

        if param_hash in self.minimax_sign_coeffs_cache:
            #If coefficients have been generated before, retrieve and return them
            cached_coeffs_list = self.minimax_sign_coeffs_cache[param_hash]
            flat_coeffs = [coeff for poly in cached_coeffs_list for coeff in poly]
            return flat_coeffs

        interval_start = -1.0
        interval_end = 1.0
        num_points = 1 << (prec.bit_length() + 1)
        x_coords = np.linspace(interval_start, interval_end, num_points)
        y_coords = np.sign(x_coords)

        #The Lattigo function generates a composite polynomial
        #x is passed through a series of polynomials
        
        generated_coeffs_list = []
        max_degree = max(degrees)
        weights = np.ones_like(x_coords)
        weights[np.abs(x_coords) < 0.05] = 100 #Heavily weigh points near zero
        
        cheb_coeffs = np.polynomial.chebyshev.chebfit(x_coords, y_coords, max_degree, w=weights)
        final_poly_coeffs = np.polynomial.chebyshev.cheb2poly(cheb_coeffs) # Convert to standard basis

        all_polys = [final_poly_coeffs.tolist()]
        for i in range(len(all_polys[-1])):
            all_polys[-1][i] /= 2.0
        all_polys[-1][0] += 0.5

        self.minimax_sign_coeffs_cache[param_hash] = all_polys
        flat_coeffs = [coeff for poly in all_polys for coeff in poly]
        
        return flat_coeffs


    #setup_lt_evaluator
    def setup_lt_evaluator():
        #currently set up at python level, so no backend binding necassary

    def NewLinearTransformEvaluator(self):
        self.linear_transforms = []
    
    def GenerateLinearTransform(self, diags_idxs, diags_data, level, bsgs_ratio=1.0, io_mode="none"):
        num_slots = self.GetPlaintextSlots(self.Encode([0.0])) # A bit of a hack to get N/2
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
        plan = self.linear_transforms[transform_id]
        diagonals = plan['diagonals']
        input_scale = self.GetCiphertextScale(ctxt_in_handle)
        input_level = self.GetCiphertextLevel(ctxt_in_handle)
        zero_ptxt = self.Encode([0.0] * self.GetCiphertextSlots(ctxt_in_handle), level=input_level, scale=input_scale)
        accumulator_ctxt = self.Encrypt(zero_ptxt)
        self.DeletePlaintext(zero_ptxt)

        #loop through each diagonal, perform Rotate->Mul->Add, and accumulate
        for diag_idx, diag_coeffs in diagonals.items():
            rotated_ctxt = self.Rotate(ctxt_in_handle, diag_idx)
            #encode the plaintext diagonal - it must match the rotated ciphertext parameters
            diag_ptxt = self.Encode(diag_coeffs, level=self.GetCiphertextLevel(rotated_ctxt), scale=self.GetCiphertextScale(rotated_ctxt))
            #multiply the rotated ciphertext by the plaintext diagonal
            term_ctxt = self.MulPlain(rotated_ctxt, diag_ptxt)

            #running total
            self.Add_Inplace(accumulator_ctxt, term_ctxt)

            #clean up
            self.DeletePlaintext(diag_ptxt)
            self.DeleteCiphertext(term_ctxt)
            if diag_idx != 0:
                self.DeleteCiphertext(rotated_ctxt)
        return accumulator_ctxt

    def DeleteLinearTransform(self, transform_id):
        if 0 <= transform_id < len(self.linear_transforms):
            self.linear_transforms[transform_id] = None
    

    def GetLinearTransformRotationKeys(self, transform_id):
        #inspects a compiled transform and returns the list of required rotation indices

        plan = self.linear_transforms[transform_id]
        #the diagonal indices are the rotation indices needed
        #rotation by 0 is a conjugation, requires specific galois key
        #Lattigo convention represent this with the Galois element for N+1
        #but we use a simple mapping: rotation_step=0 -> galois_elt=0 (placeholder)
        required_rotations = list(plan['diagonals'].keys())
        if 0 in required_rotations:
            #for conjugation
            pass
        return required_rotations

    def GenerateLinearTransformRotationKey(self, rotation_step):
        #Generates a single Galois key for a given rotation and caches
        galois_key_handle = self.CreateGaloisKey_With_Shifts([rotation_step])
        self.GenerateGaloisKey(galois_key_handle)
        self.rotation_keys_cache[rotation_step] = galois_key_handle

    def GenerateAndSerializeRotationKey(self, rotation_step):
        #Generates a specific Galois key and returns its serialized byte representation
        galois_key_handle = self.CreateGaloisKey_With_Shifts([rotation_step])
        self.GenerateGaloisKey(galois_key_handle)
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

    def setup_bootstrapper():
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
            self.lib.HEonGPU_CKKS_ArithmeticOperator_GenerateBootstrappingParams,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(ctypes.POINTER(c_int)),
                ctypes.POINTER(c_uint64)
            ],
            restype=ctypes.c_int 
        )
        self._RegularBootstrapping = HEonGPUFunction(
            self.lib.HEonGPU_CKKS_ArithmeticOperator_GenerateBootstrappingParams,
            argtypes=[
                ctypes.POINTER(HE_CKKS_ArithmeticOperator),
                ctypes.POINTER(HE_CKKS_Ciphertext),
                ctypes.POINTER(HE_CKKS_GaloisKey),
                ctypes.POINTER(HE_CKKS_RelinKey),
                ctypes.POINTER(C_ExecutionOptions)
            ],
            restype=ctypes.POINTER(HE_CKKS_Ciphertext)
        )
    

    BOOTSTRAP_PRESET_CONFIG = {
        # Target Depth: { 'taylor_number', 'CtoS_piece', 'StoC_piece', 'less_key_mode' }
        1: {
            'taylor_number': 7, 
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        2: {
            'taylor_number': 7,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        3: {
            'taylor_number': 7,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        4: {
            'taylor_number': 11,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        5: {
            'taylor_number': 11,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        6: {
            'taylor_number': 15,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        7: {
            'taylor_number': 15,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        8: {
            'taylor_number': 15,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        9: {
            'taylor_number': 15, # WARNING: Max recommended value reached
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        },
        10: {
            'taylor_number': 15,
            'CtoS_piece': 8,
            'StoC_piece': 8,
            'less_key_mode': False
        }
    }
    def NewBootstrapper(self, logPs, length, num_slots):
        #we dont care about the exaxt logPs, or num_slots, just the length. HEonGPU will create the logPs from the parameters given:


        config_params = BOOTSTRAP_PRESET_CONFIG[length]

        boot_config = C_BootstrappingConfig(
            CtoS_piece=config_params['CtoS_piece'],
            StoC_piece=config_params['StoC_piece'],
            taylor_number=config_params['taylor_number'],
            less_key_mode=config_params['less_key_mode']
        )


        self.bootstrap_handle = self._GenerateBootstrappingParams(
            self.arithmeticoperator_handle,
            ctypes.c_double(self.scale),
            ctypes.byref(boot_config)
        )

    def Bootstrap(self, ct, num_slots):
        indices_ptr = ctypes.POINTER(ctypes.c_int)()
        count = ctypes.c_size_t()
        
        status = self._GetBootstrappingKeyIndices(
            self.arithmeticoperator_handle,
            ctypes.byref(indices_ptr),
            ctypes.byref(count)
        )
        galois_key_handle = self._CreateGaloisKeyWithIndices(
            self.context_handle,
            indices_ptr,
            count
        )
        bootstrapped_ct = self._RegularBootstrapping(
            self.arithmeticoperator_handle,
            ct,
            galois_key_handle,
            self.relinkey_handle,
            None 
        )

    
    






    

