import sys
import math
class PlainTensor:
    def __init__(self, scheme, ptxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend
        self.encoder = scheme.encoder

        self.values = ptxt_ids


        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        self.shape = shape 
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeletePlaintext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        return str(self.decode())
    
    def mul(self, other, in_place=False):
        if isinstance(other, CipherTensor):
            mul_id = self.evaluator.mul(self.values, other.values, in_place)
            if in_place:
                return self
            else:
                return CipherTensor(self.evaluator, self.encoder, self.encryptor, mul_id, self.shape)
        elif isinstance(other, (int, float)):
            mul_id = self.evaluator.mul_scalar(self.values, other, in_place)
            if in_place:
                return self
            else:
                return CipherTensor(self.evaluator, self.encoder, self.encryptor, mul_id, self.shape)
        # This is the new logic that handles a raw plaintext object
        elif isinstance(other, ctypes.POINTER(HE_CKKS_Ciphertext)):
            mul_id = self.evaluator.mul_plain(self.values, other, in_place)
            if in_place:
                return self
            else:
                return CipherTensor(self.evaluator, self.encoder, self.encryptor, mul_id, self.shape)
        else:
            raise ValueError(f"Multiplication between CipherTensor and "
                            f"{type(other)} is not supported.")


    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def _check_valid(self, other):
        return 
        
    def get_ids(self):
        return self.ids
    
    def scale(self):
        return self.backend.GetPlaintextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ptxt in self.ids:
            self.backend.SetPlaintextScale(ptxt, scale)

    def level(self):
        return self.backend.GetPlaintextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetPlaintextSlots(self.ids[0])
    
    def min(self):
        return self.decode().min()
    
    def max(self):
        return self.decode().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def decode(self):
        return self.encoder.decode(self)
    

class CipherTensor:
    def __init__(self, scheme, ctxt_ids, shape, on_shape=None):
        self.scheme = scheme
        self.backend = scheme.backend 
        self.encryptor = scheme.encryptor
        self.evaluator = scheme.evaluator
        self.bootstrapper = scheme.bootstrapper

        self.values = ctxt_ids

        if not isinstance(ctxt_ids, list):
            self.ids = [ctxt_ids]
        else:
            self.ids = ctxt_ids
            
        self.shape = shape 
        self.on_shape = on_shape or shape


    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeleteCiphertext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        ptxt = self.decrypt()
        return str(ptxt.decode())
    
    #--------------#
    #  Operations  #
    #--------------#
    
    def __neg__(self):
        neg_ids = []
        for ctxt in self.ids:
            neg_id = self.evaluator.negate(ctxt)
            neg_ids.append(neg_id)

        return CipherTensor(self.scheme, neg_ids, self.shape, self.on_shape)
    
    def add(self, other, in_place=False):
        self._check_valid(other)

        add_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                add_id = self.evaluator.add_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                add_id = self.evaluator.add_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                add_id = self.evaluator.add_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Addition between CipherTensor and "
                                 f"{type(other)} is not supported.")

            add_ids.append(add_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, add_ids, self.shape, self.on_shape)
    
    def __add__(self, other):
        return self.add(other, in_place=False)

    def __iadd__(self, other):
        return self.add(other, in_place=True)
    
    def sub(self, other, in_place=False):
        self._check_valid(other)

        sub_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                sub_id = self.evaluator.sub_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                sub_id = self.evaluator.sub_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                sub_id = self.evaluator.sub_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Subtraction between CipherTensor and "
                                 f"{type(other)} is not supported.")

            sub_ids.append(sub_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, sub_ids, self.shape, self.on_shape)
    
    def __sub__(self, other):
        return self.sub(other, in_place=False)

    def __isub__(self, other):
        return self.sub(other, in_place=True)
    
    def mul(self, other, in_place=False):
        self._check_valid(other)

        mul_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                mul_id = self.evaluator.mul_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                mul_id = self.evaluator.mul_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                mul_id = self.evaluator.mul_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Multiplication between CipherTensor and "
                                 f"{type(other)} is not supported.")
            
            mul_ids.append(mul_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 
    
    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def mod_drop(self, in_place=True):
        if not in_place:
             # Create a clone for out-of-place operation
            new_ids = [self.backend.CloneCiphertext(cid) for cid in self.ids]
            new_tensor = CipherTensor(self.scheme, new_ids, self.shape, self.on_shape)
            for cid in new_tensor.ids:
                self.evaluator.mod_drop(cid, in_place=True) # in-place on the clone
            return new_tensor
        else:
            # In-place operation on the current tensor
            for cid in self.ids:
                self.evaluator.mod_drop(cid, in_place=True)
            return self


    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def roll(self, shift, in_place=False):
        if not isinstance(shift, int):
            raise TypeError(f"Roll amount must be an integer, not {type(shift)}")

        ctxt_handle = self.values[0] if isinstance(self.values, list) else self.values

        if in_place:
            self.evaluator.rotate(ctxt_handle, shift, in_place=True)
            return self
        else:
            rot_id = self.evaluator.rotate(ctxt_handle, shift, in_place=False)
            final_handle = rot_id[0] if isinstance(rot_id, list) else rot_id
            
            # This is the corrected constructor call matching your code.
            return CipherTensor(self.scheme, final_handle, self.shape)


    
    def _check_valid(self, other):
        return
    
    #----------------------
    #
    #---------------------
    
    def scale(self):
        return self.backend.GetCiphertextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ctxt in self.ids:
            self.backend.SetCiphertextScale(ctxt, scale)

    def level(self):
        return self.backend.GetCiphertextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetCiphertextSlots(self.ids[0])
    
    def degree(self):
        return self.backend.GetCiphertextDegree(self.ids[0])
    
    def min(self):
        return self.decrypt().min()
    
    def max(self):
        return self.decrypt().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots)) # sparse bootstrapping
        
        btp_ids = []
        for ctxt in self.ids:
            btp_id = self.bootstrapper.bootstrap(ctxt, slots)
            btp_ids.append(btp_id)

        return CipherTensor(self.scheme, btp_ids, self.shape, self.on_shape)
        
    def decrypt(self):
        return self.encryptor.decrypt(self)