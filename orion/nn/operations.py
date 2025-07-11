import math
import torch
import ctypes
from .module import Module, timer
from orion.backend.python.tensors import PlainTensor, CipherTensor
class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x, y):
        return x * y
    

class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min 
        self.input_max = input_max 
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (self.margin * half_range)).item()
        self.high = (center + (self.margin * half_range)).item()

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # We'll then encode the prescale at the level of the input ciphertext
        # to ensure its rescaling is errorless
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = self.scheme.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = self.scheme.encoder.encode(
            prescale_vec, level=self.input_level, scale=ql)

    @timer
    def forward(self, x):
        #Hijak this function for a test:

        print("\n[DEBUG] --- RUNNING ISOLATED BOOTSTRAP TEST ---")
        backend = x.evaluator.backend
        total_levels = len(self.scheme.params.get_logq())
        
        # Handles we need to create and manually clean up
        pt_handle = None
        ct_handle = None
        ptxt_mult_handle = None
        bootstrapped_ct = None
        decrypted_pt = None

        
        # === Step 1: Create a new Ciphertext from scratch ===
        print("[DEBUG] Step 1: Creating a new ciphertext from scratch.")
        num_slots = backend.GetCiphertextSlots(x)
        message = [1.0] * num_slots
        
        # Create and encode a plaintext
        pt_handle = backend.CreatePlaintext()
        data_to_encode = (ctypes.c_double * len(message))(*message)
        backend._Encode(backend.encoder_handle, pt_handle, data_to_encode, len(message), backend.scale, None)

        # Create and encrypt into a ciphertext
        ct_handle = backend.CreateCiphertext()
        ct_handle = backend.Encrypt(pt_handle) 
        
        # Wrap the handle in a high-level CipherTensor for easy use
        test_ct = CipherTensor(self.scheme, [ct_handle], x.shape, x.on_shape)
        print("[DEBUG]   - Successfully created a fresh ciphertext.")

        # === Step 2: Multiply the new ciphertext ===
        print("[DEBUG] Step 2: Multiplying by a plaintext of '2.0'.")
        ptxt_mult_handle = backend.CreatePlaintext()
        scale = backend.GetCiphertextScale(test_ct.ids[0])
        mult_data = (ctypes.c_double * 1)(2.0)
        backend._Encode(backend.encoder_handle, ptxt_mult_handle, mult_data, 1, scale, None)
        
        test_ct.evaluator.mul_plaintext(test_ct.ids[0], ptxt_mult_handle, in_place=True)
        # The rescale is implicit in mul_plaintext, so one level is dropped here.
        print("[DEBUG]   - Multiplication and implicit rescale complete.")
        
        # === Step 3: Drop to required level for bootstrap ===
        print("[DEBUG] Step 3: Dropping to required level for bootstrap.")
        required_level = 1
        while True:
            current_depth = backend.GetCiphertextDepth(test_ct.ids[0])
            current_level = total_levels - current_depth
            if current_level <= required_level:
                break
            
            print(f"    - Current level is {current_level}. Dropping level via manual rescale.")
            backend.ModDropCiphertextInplace(test_ct.ids[0])

        # === Step 4: Bootstrap the ciphertext ===
        final_depth = backend.GetCiphertextDepth(test_ct.ids[0])
        final_level = total_levels - final_depth
        final_scale = backend.GetCiphertextScale(test_ct.ids[0])

        print(f"\n[DEBUG] Pre-Bootstrap Check on 'test_ct':")
        print(f"    - Final depth: {final_depth}")
        print(f"    - Final Level: {final_level}")
        print(f"    - Final Scale: {final_scale:.4e}\n")
        print(f"[DEBUG] Step 4: Ciphertext is at depth {final_depth}. Calling bootstrap...")
        print(f"    - Type BEFORE bootstrap call: {type(test_ct)}")
        bootstrapped_ct = test_ct.bootstrap()
        print(f"    - Type AFTER bootstrap call: {type(bootstrapped_ct)}")

        # === Step 5: Decrypt and decode the final result ===
        print("[DEBUG] Step 5: Decrypting and decoding the result.")
        decrypted_pt = bootstrapped_ct.decrypt()
        decoded_values = decrypted_pt.decode()

        print("\n[SUCCESS] Isolated bootstrap test completed without crashing.")
        print(f"    - Final Decoded values (first 5): {decoded_values[:5]}")


        # --- IMPORTANT: Clean up all allocated FHE objects ---
        print("[DEBUG] Cleaning up all test handles.")
        if pt_handle: backend.DeletePlaintext(pt_handle)
        if ct_handle: backend.DeleteCiphertext(ct_handle) # This handle is now inside test_ct
        if ptxt_mult_handle: backend.DeletePlaintext(ptxt_mult_handle)
        if bootstrapped_ct: bootstrapped_ct.delete() # Assumes a delete method on CipherTensor
        if decrypted_pt: decrypted_pt.delete() # Assumes a delete method on PlainTensor


        print("--- ISOLATED TEST FINISHED ---")

        x = 1/0





        if not self.he_mode:
            return x
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This let's us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2).
        if self.constant != 0:
            x += self.constant
        x *= self.prescale_ptxt
 
        x = x.bootstrap()

        # Scale and shift back to the original range
        if self.postscale != 1:
            x *= self.postscale 
        if self.constant != 0:
            x -= self.constant

        return x












