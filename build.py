from distutils.core import setup, Extension
from Cython.Distutils import build_ext

cc_flags = ['-O3', '-march=native', '-mtune=native']

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "modified_dubins", ["modified_dubins.pyx"],
            extra_compile_args=cc_flags, extra_link_args=cc_flags
        ),
    ],
)