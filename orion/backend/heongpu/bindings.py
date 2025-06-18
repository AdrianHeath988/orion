import os 
import ctypes
import platform

import torch
import numpy as np


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
        # self.setup_poly_evaluator()
        # self.setup_lt_evaluator()
        # self.setup_bootstrapper()

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

        #TODO: Determine what the argtypes of encode mean, and add wrapper function
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
    #TODO: This, but must first complete encode
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


    
    






    

