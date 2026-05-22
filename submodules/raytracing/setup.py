import os
import urllib.request
import tarfile
import re
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

def find_eigen(min_ver=(3, 3, 0)):
    try_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        os.path.expanduser('~/.local/include/eigen3'),
        'C:/Program Files/eigen3',
        'C:/Program Files (x86)/eigen3',
    ]
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        try_paths.append(os.path.join(conda_prefix, 'include', 'eigen3'))
        try_paths.append(os.path.join(conda_prefix, 'include'))

    min_ver_str = '.'.join(map(str, min_ver))
    EIGEN_WEB_URL = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2'
    TMP_EIGEN_FILE = 'tmp_eigen.tar.bz2'
    TMP_EIGEN_DIR = 'eigen-3.3.7'

    eigen_path = None
    for path in try_paths:
        macros_path = os.path.join(path, 'Eigen/src/Core/util/Macros.h')
        if os.path.exists(macros_path):
            with open(macros_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            world = re.search(r'#define\s+EIGEN_WORLD_VERSION\s+(\d+)', content)
            major = re.search(r'#define\s+EIGEN_MAJOR_VERSION\s+(\d+)', content)
            minor = re.search(r'#define\s+EIGEN_MINOR_VERSION\s+(\d+)', content)
            
            if not world or not major or not minor:
                print('Failed to parse macros file, using path anyway:', path)
                eigen_path = path
                break
            
            ver = (int(world.group(1)), int(major.group(1)), int(minor.group(1)))
            ver_str = '.'.join(map(str, ver))
            if ver < min_ver:
                print('Found unsuitable Eigen version', ver_str, 'at', path, '(need >= ' + min_ver_str + ')')
            else:
                print('Found Eigen version', ver_str, 'at', path)
                eigen_path = path
                break

    if eigen_path is None:
        try:
            print("Couldn't find Eigen locally, downloading...")
            req = urllib.request.Request(
                EIGEN_WEB_URL,
                data=None,
                headers={
                    'User-Agent':
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
                })

            with urllib.request.urlopen(req) as resp,\
                 open(TMP_EIGEN_FILE, 'wb') as file:
                data = resp.read()
                file.write(data)
            
            tar = tarfile.open(TMP_EIGEN_FILE)
            tar.extractall()
            tar.close()

            eigen_path = os.path.join(_src_path, TMP_EIGEN_DIR)
            os.remove(TMP_EIGEN_FILE)
        except:
            print('Download failed, failed to find Eigen')

    if eigen_path is not None:
        print('Found eigen at', eigen_path)

    return eigen_path

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

setup(
    name='raytracing',
    version='0.1.0',
    description='CUDA RayTracer with BVH acceleration',
    url='https://github.com/ashawkey/raytracing',
    author='kiui',
    author_email='ashawkey1999@gmail.com',
    ext_modules=[
        CUDAExtension(
            name='_raytracing',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'raytracer.cu',
                'bindings.cpp',
            ]],
            include_dirs=[
                os.path.join(_src_path, 'include'),
                find_eigen(),
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
    install_requires=[
        'ninja',
        'trimesh',
        'opencv-python',
        'torch',
        'numpy',
        'tqdm',
        'matplotlib',
        'dearpygui',
    ],
)
