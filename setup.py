from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# cc_flags = ['-O3', '-march=native', '-mtune=native']
cc_flags = ['-O2']

modules = cythonize([
    Extension(
        "modified_dubins", ["modified_dubins.pyx"],
        extra_compile_args=cc_flags,
        extra_link_args=cc_flags
    ),
])

setup(
    name='modified-3d-dubins',
    version='1.0',
    description='A modified version of Dubin\'s path based on altitude modifications and vector algebra',
    author='Tolga Demirdal',
    url='https://github.com/shadymeowy/modified-3d-dubins',
    setup_requires=["cython"],
    install_requires=[],
    ext_modules=modules
)
