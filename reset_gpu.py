import ctypes

try:
    cuda = ctypes.CDLL('libcudart.so')
    print("INFO: Successfully loaded CUDA runtime library.")

    print("INFO: Attempting to reset CUDA device...")
    result = cuda.cudaDeviceReset()

    if result == 0:
        print("SUCCESS: cudaDeviceReset() completed successfully!")
    else:
        print(f"ERROR: cudaDeviceReset() failed with error code: {result}")

except Exception as e:
    print(f"ERROR: An error occurred: {e}")

