from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension



setup(name='int_quantization',
      ext_modules=[CUDAExtension('int_quantization', ['int_quantization.cpp',
                                                      'gemmlowp.cu'
                                              ])],
      cmdclass={'build_ext': BuildExtension})

# for installation execute:
# > python build_int_quantization.py install
# record list of all installed files:
# > python build_int_quantization.py install --record files.txt
