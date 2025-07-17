import time
import math
import torch
import orion
import orion.models as models
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
bindings_dir = os.path.join(project_dir, 'orion', 'backend', 'heongpu')
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

from bindings import get_heongpu_library


from orion.core.utils import (
    get_mnist_datasets,
    mae, 
    train_on_mnist
)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/lola2.yml")
trainloader, testloader = get_mnist_datasets(data_dir="../data", batch_size=1)
net = models.LoLA()

# Train model (optional)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_mnist(net, data_dir="../data", epochs=1, device=device)

# Get a test batch to pass through our network
inp, _ = next(iter(testloader))

# Run cleartext inference
net.eval()
out_clear = net(inp)
print(f"[DEBUG] out_clear: {out_clear[:10]}")
# Prepare for FHE inference. 
# Certain polynomial activation functions require us to know the precise range
# of possible input values. We'll determine these ranges by aggregating
# statistics from the training set and applying a tolerance factor = margin.
orion.fit(net, trainloader)
print("\nStarting Compiling", flush=True)
input_level = orion.compile(net)

# Encode and encrypt the input vector 
print("\nStarting Encoding", flush=True)
print(inp)
print(input_level)
vec_ptxt = orion.encode(inp, input_level)
check = vec_ptxt.decode()
print("After Encoding")
print(check)
print("\nStarting Encrypting", flush=True)
vec_ctxt = orion.encrypt(vec_ptxt)
check_pt = vec_ctxt.decrypt()
check = check_pt.decode()
print("After encrypting:")
print(check)
print("\nStarting net.he", flush=True)
net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)


end = time.time()
print("\Finished FHE inference", flush=True)
# Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# Compare the cleartext and FHE results.
print()
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
torch.cuda.synchronize()
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")
print(f"[DEBUG] FINISHED Example")
get_heongpu_library().shutdown()




print("HEonGPU resources released. Program will now exit cleanly.")
sys.exit(0)

