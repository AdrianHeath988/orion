import sys
import os
import platform
import subprocess
from pathlib import Path

# --- Lattigo Build Function ---
def build_lattigo(root_dir, env):
    print("=== Building Lattigo Go shared library ===")
    
    # Determine the output filename based on platform
    if platform.system() == "Windows":
        output_file = "lattigo-windows.dll"
    elif platform.system() == "Darwin":  # macOS
        if platform.machine().lower() in ("arm64", "aarch64"):
            output_file = "lattigo-mac-arm64.dylib"
        else:
            output_file = "lattigo-mac.dylib"
    elif platform.system() == "Linux":
        output_file = "lattigo-linux.so"
    else:
        raise RuntimeError(f"Unsupported platform for Lattigo: {platform.system()}")
    
    # Set up Lattigo paths
    lattigo_backend_dir = root_dir / "orion" / "backend" / "lattigo"
    lattigo_output_path = lattigo_backend_dir / output_file
    
    # Lattigo Build command
    lattigo_build_cmd = [
        "go", "build", 
        "-buildmode=c-shared",
        "-buildvcs=false", 
        "-o", str(lattigo_output_path),
        str(lattigo_backend_dir)
    ]
    
    # Run the Lattigo build command
    try:
        print(f"Running: {' '.join(lattigo_build_cmd)}")
        # For Go build, stdout/stderr can be useful on error but often verbose on success
        process = subprocess.run(lattigo_build_cmd, cwd=str(lattigo_backend_dir), env=env, check=True, capture_output=True, text=True)
        if process.stdout:
            print(f"Lattigo Go build stdout:\n{process.stdout}")
        if process.stderr: # Go often prints to stderr even on success for warnings/status
            print(f"Lattigo Go build stderr:\n{process.stderr}")
        print(f"Successfully built {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Lattigo Go build failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Lattigo Go build stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Lattigo Go build stderr:\n{e.stderr}")
        sys.exit(1)

# --- HEonGPU Build Function ---
def build_heongpu(root_dir, env):
    print("\n=== Building HEonGPU C++/CUDA library ===")
    heongpu_source_dir = root_dir / "orion" / "backend" / "heongpu" / "adrianHEonGPU"
    heongpu_build_dir_name = "build_heongpu" # Using a distinct build directory name
    heongpu_build_dir = heongpu_source_dir / heongpu_build_dir_name
    if not heongpu_source_dir.is_dir():
        print(f"HEonGPU source directory not found: {heongpu_source_dir}")
        print("Please ensure the HEonGPU submodule is initialized and at the correct path.")
        sys.exit(1)

    # 1. CMake Configure
    # Using a build directory inside heongpu_source_dir named "build_heongpu"
    cmake_configure_cmd = [
        "cmake",
        "-S", ".",  # Source directory (current dir when cwd is heongpu_source_dir)
        "-B", heongpu_build_dir_name, # Build directory
        "-D", "CMAKE_BUILD_TYPE=Debug",
        "-D", "HEonGPU_BUILD_EXAMPLES=ON",
        "-D", "CMAKE_CUDA_ARCHITECTURES=86"
        # Add other -D flags if needed, e.g., CMAKE_BUILD_TYPE
        # Consider adding -D CMAKE_INSTALL_PREFIX=./install_heongpu for local install
    ]
    try:
        print(f"Running CMake configure for HEonGPU: {' '.join(str(c) for c in cmake_configure_cmd)}")
        print(f"Working directory: {heongpu_source_dir}")
        process = subprocess.run(cmake_configure_cmd, cwd=str(heongpu_source_dir), env=env, check=True, capture_output=True, text=True)
        if process.stdout:
            print(f"HEonGPU CMake configure stdout:\n{process.stdout}")
        if process.stderr:
            print(f"HEonGPU CMake configure stderr:\n{process.stderr}")
        print("HEonGPU CMake configuration successful.")
    except subprocess.CalledProcessError as e:
        print(f"HEonGPU CMake configure failed with exit code {e.returncode}")
        if e.stdout:
            print(f"HEonGPU CMake configure stdout:\n{e.stdout}")
        if e.stderr:
            print(f"HEonGPU CMake configure stderr:\n{e.stderr}")
        sys.exit(1)

    # 2. CMake Build
    cmake_build_cmd = [
        "cmake",
        "--build", str(heongpu_build_dir) # Path to the build directory
    ]
    try:
        print(f"Running CMake build for HEonGPU: {' '.join(str(c) for c in cmake_build_cmd)}")
        process = subprocess.run(cmake_build_cmd, env=env, check=True, capture_output=True, text=True)
        if process.stdout: # CMake build can be very verbose
            print(f"HEonGPU CMake build stdout (summary):\n{process.stdout[:1000]}...") # Print first 1000 chars
        if process.stderr:
            print(f"HEonGPU CMake build stderr:\n{process.stderr}")
        print("HEonGPU CMake build successful.")
    except subprocess.CalledProcessError as e:
        print(f"HEonGPU CMake build failed with exit code {e.returncode}")
        if e.stdout:
            print(f"HEonGPU CMake build stdout:\n{e.stdout}")
        if e.stderr:
            print(f"HEonGPU CMake build stderr:\n{e.stderr}")
        sys.exit(1)

def build_openfhe(root_dir, env):
    """
    Builds the OpenFHE C++ library using CMake and Make.
    """
    print("\n=== Building OpenFHE C++ library ===")
    
    # 1. Set up paths for OpenFHE
    # Path based on your project structure
    openfhe_source_dir = root_dir / "orion" / "backend" / "openfhe" / "openfhe-development"
    openfhe_build_dir = openfhe_source_dir / "build"
    openfhe_install_dir = openfhe_source_dir / "install" # Local install directory

    # Check if the source directory exists
    if not openfhe_source_dir.is_dir():
        print(f"OpenFHE source directory not found: {openfhe_source_dir}")
        print("Please ensure the OpenFHE submodule is initialized and at the correct path.")
        sys.exit(1)

    # Create the build directory, similar to 'mkdir build' [cite: 35]
    openfhe_build_dir.mkdir(exist_ok=True)

    # 2. CMake Configure step
    # This command is equivalent to running 'cmake ..' from the build directory [cite: 37]
    # We add -DCMAKE_INSTALL_PREFIX to install locally and avoid needing sudo 
    cmake_configure_cmd = [
        "cmake",
        "..",
        f"-DCMAKE_INSTALL_PREFIX={openfhe_install_dir}",
        # You can add other CMake flags here, e.g., -DBUILD_EXAMPLES=OFF
    ]
    try:
        print(f"Running CMake configure for OpenFHE: {' '.join(str(c) for c in cmake_configure_cmd)}")
        print(f"Working directory: {openfhe_build_dir}")
        process = subprocess.run(cmake_configure_cmd, cwd=str(openfhe_build_dir), env=env, check=True, capture_output=True, text=True)
        if process.stdout:
            print(f"OpenFHE CMake configure stdout:\n{process.stdout}")
        if process.stderr:
            print(f"OpenFHE CMake configure stderr:\n{process.stderr}")
        print("OpenFHE CMake configuration successful.")
    except subprocess.CalledProcessError as e:
        print(f"OpenFHE CMake configure failed with exit code {e.returncode}")
        if e.stdout:
            print(f"OpenFHE CMake configure stdout:\n{e.stdout}")
        if e.stderr:
            print(f"OpenFHE CMake configure stderr:\n{e.stderr}")
        sys.exit(1)


# --- Main Build Function Called by Poetry ---
def build(setup_kwargs=None):
    root_dir = Path(__file__).parent.parent
    
    # Prepare environment (copying OS environment)
    # CGO_ENABLED and GOARCH are specific to Lattigo's Go build.
    # CMake/HEonGPU might need other env vars (e.g., for CUDA),
    # ensure they are set in your shell environment if needed.
    env = os.environ.copy()
    env["CGO_ENABLED"] = "1" # For Lattigo
    if platform.system() == "Darwin": # For Lattigo
        if platform.machine().lower() in ("arm64", "aarch64"):
            env["GOARCH"] = "arm64"
        else:
            env["GOARCH"] = "amd64"

    # Build Lattigo
    build_lattigo(root_dir, env.copy()) # Pass a copy of env in case functions modify it
    # Build HEonGPU
    # HEonGPU doesn't need GOARCH, CGO_ENABLED from Lattigo's specific env settings.
    # It will use the general os.environ copy, unless specific env vars are needed for CMake/CUDA.
    build_heongpu(root_dir, os.environ.copy())
    build_openfhe(root_dir, os.environ.copy())
    print("\n=== All backend builds completed (or skipped if chosen) ===")
    
    # Return setup_kwargs for Poetry (important)
    return setup_kwargs or {}

# --- Script execution for direct call (e.g., for testing) ---
if __name__ == "__main__":
    print("Running build_backends.py directly for testing...")
    build()
    print("Direct script execution finished.")
    # Note: sys.exit() was removed from here because the build() function
    # now calls sys.exit() on failure, and poetry expects a return value or exception.