import os
import subprocess
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

class CMakeExtensionBuild(BuildExtension):
    def run(self):
        build_directory = os.path.join(_src_path, 'build')
        os.makedirs(build_directory, exist_ok=True)
        
        cmake_args = ['cmake', '..']
        
        cc = os.environ.get('CC')
        cxx = os.environ.get('CXX')
        if cc:
            cmake_args.append(f'-DCMAKE_C_COMPILER={cc}')
            cmake_args.append(f'-DCMAKE_CUDA_HOST_COMPILER={cc}')
        if cxx:
            cmake_args.append(f'-DCMAKE_CXX_COMPILER={cxx}')
            
        subprocess.check_call(cmake_args, cwd=build_directory)
        subprocess.check_call(['make', f'-j{os.cpu_count()}'], cwd=build_directory)
        super().run()

nvcc_flags = [
    '-O3', '-std=c++17',
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path

include_directories = [
    os.path.join(_src_path, 'include'),
    os.path.join(_src_path, "build"),
    os.path.join(_src_path, 'include', 'optix'),
    os.path.join(_src_path, 'include', 'glm'),
]

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    include_directories.append(os.path.join(conda_prefix, 'include'))

setup(
    name='surfel_tracer',
    version='0.1.0',
    description='2D Gaussian RayTracer',
    author='Chun Gu',
    author_email='cgu19@fudan.edu.cn',
    packages=['surfel_tracer'],
    ext_modules=[
        CUDAExtension(
            name='surfel_tracer._C',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'bindings.cu',
            ]],
            include_dirs=include_directories,
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': CMakeExtensionBuild,
    },
    install_requires=[
        'ninja',
        'trimesh',
        'opencv-python',
        'torch',
        'numpy',
        'tqdm',
        'dearpygui',
    ],
)
