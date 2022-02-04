from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension#, CUDAExtension

setup(
      name='sgd_cpp',
      ext_modules=[
            CppExtension(
                  name='sgd_cpu',
                  sources=['sgd_binding.cpp'],
                  extra_compile_args = ["-std=c++17"], 
            ),
            # CUDAExtension(...),
      ],
      cmdclass={'build_ext': BuildExtension}
      )
