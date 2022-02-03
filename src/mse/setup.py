from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension#, CUDAExtension

setup(
      name='mse_cpp',
      ext_modules=[
            CppExtension(
                  name='mse_cpu',
                  sources=['mse_binding.cpp'],
            ),
            # CUDAExtension(...),
      ],
      cmdclass={'build_ext': BuildExtension}
      )
