from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='lltm',
    ext_modules=[
        CppExtension('lltm', ['lltm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })