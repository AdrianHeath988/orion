# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext # Rename original


# Default to 'lattigo' if not specified
ORION_BACKEND_TO_BUILD = os.environ.get('ORION_BACKEND', 'lattigo').lower()
print(f"--- Building Orion with backend: {ORION_BACKEND_TO_BUILD} ---")

packages = [
    'orion',
    'orion.backend',
    # Conditionally include backend Python code packages
    # 'orion.backend.lattigo', # Included if backend == 'lattigo'
    # 'orion.backend.heongpu', # Included if backend == 'heongpu'
    'orion.backend.python',
    'orion.core',
    'orion.models',
    'orion.nn'
]
# Add backend-specific package based on selection
if ORION_BACKEND_TO_BUILD == 'lattigo':
    packages.append('orion.backend.lattigo')
elif ORION_BACKEND_TO_BUILD == 'heongpu':
    packages.append('orion.backend.heongpu')
# Add other backends here 


package_data = {
    '': ['*.yml', '*.md'], 
}
if ORION_BACKEND_TO_BUILD == 'lattigo':
    package_data['orion.backend.lattigo'] = ['*.so', '*.dylib', '*.dll', '*.h']
elif ORION_BACKEND_TO_BUILD == 'heongpu':
     package_data['orion.backend.heongpu'] = ['*.so', '*.dylib', '*.dll', '*.h'] # Adjust as needed

# --- Dependencies ---
install_requires = [
    'PyYAML>=6.0',
    'certifi>=2024.2.2',
    'h5py>=3.5.0',
    'matplotlib>=3.1.0',
    'numpy>=1.21.0',
    'scipy>=1.7.0,<=1.14.1',
    'torch>=2.2.0',
    'torchvision>=0.17.0',
    'tqdm>=4.30.0'
]


class build_ext(_build_ext):
    def run(self):
        if ORION_BACKEND_TO_BUILD == 'lattigo':
            # --- Lattigo Build ---
            print("*"*10, "Building Lattigo Backend", "*"*10)
            try:
                from tools.build_lattigo import build
                build() # Call the Go build process
                print("*"*10, "Finished Building Lattigo Backend", "*"*10)
            except ImportError:
                print("ERROR: tools/build_lattigo.py not found or missing function.", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"ERROR: Lattigo build failed: {e}", file=sys.stderr)
                sys.exit(1)

        elif ORION_BACKEND_TO_BUILD == 'heongpu':
            # --- HEonGPU Build ---
            print("*"*10, "Building HEonGPU Backend", "*"*10)
            heongpu_source_dir = os.path.abspath('orion/backend/heongpu/source')
            cmake_build_dir = os.path.join(self.build_temp, 'heongpu_build')
            cmake_install_dir = os.path.abspath('orion/backend/heongpu/lib') # Install lib here?
            os.makedirs(cmake_build_dir, exist_ok=True)
            os.makedirs(cmake_install_dir, exist_ok=True)

            cmake_args = [
                f'-DCMAKE_INSTALL_PREFIX={cmake_install_dir}', # Install locally
                '-DBUILD_SHARED_LIBS=ON', # Build shared library
                '-DCMAKE_CXX_STANDARD=gnu++20',
                '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10',
                '-DCMAKE_CUDA_ARCHITECTURES=86',
                # '-DCMAKE_CUDA_COMPILER=/path/to/nvcc'
                # '-DCMAKE_BUILD_TYPE=Release',
            ]
            build_args = ['--config', 'Release']

            try:
                print(f"--- Configuring HEonGPU (source: {heongpu_source_dir}) ---")
                subprocess.check_call(['cmake', heongpu_source_dir] + cmake_args, cwd=cmake_build_dir)
                print("--- Building HEonGPU library ---")
                subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=cmake_build_dir)

                

                print("*"*10, "Finished Building HEonGPU Library", "*"*10)


            except subprocess.CalledProcessError as e:
                print(f"ERROR: HEonGPU CMake build failed: {e}", file=sys.stderr)
                sys.exit(1)
            except FileNotFoundError:
                 print(f"ERROR: CMake command not found. Please install CMake.", file=sys.stderr)
                 sys.exit(1)
            except Exception as e:
                 print(f"ERROR: HEonGPU build failed: {e}", file=sys.stderr)
                 sys.exit(1)
        _build_ext.run(self)

# --- Define Extensions (Only needed for C++/Cython/Pybind11 wrappers) ---
ext_modules = []
if ORION_BACKEND_TO_BUILD == 'heongpu':
    ext_modules.append(
        Extension(
            'orion.backend.heongpu.heongpu_backend', # name of the .so file
            sources=['orion/backend/heongpu/heongpu_wrapper.cpp'], # Wrapper source
            include_dirs=[ 
                'orion/backend/heongpu/source/include', 
            ],
            library_dirs=[ # Where to find the compiled libheongpu.so/.a
                os.path.abspath('orion/backend/heongpu/lib/lib') # Adjust path
            ],
            libraries=['heongpu'], 
            language='c++',
              
        )
    )



# --- Main Setup Arguments ---
# IDK if this needs to change?
# Copied from previous setup
setup_kwargs = {
    'name': 'orion-fhe',
    'version': '1.0.2', # Consider bumping version
    'description': 'A Fully Homomorphic Encryption Framework for Deep Learning',
    'long_description': open('README.md').read(), # Read from README.md
    'author': 'Austin Ebel',
    'author_email': 'abe5240@nyu.edu',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/baahl-nyu/orion',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.13', # Check if HEonGPU has different reqs
    'ext_modules': ext_modules, # Add extensions if any
    'cmdclass': {'build_ext': build_ext}, # Use custom build class
    'zip_safe': False, # Often needed for C extensions / package_data
}

# --- Call Setup ---
setup(**setup_kwargs)