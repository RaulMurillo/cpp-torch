from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension#, CUDAExtension

setup(
      name='linear_cpp',
      ext_modules=[
            CppExtension(
                  name='linear_cpu',
                  sources=['linear_binding.cpp'],
            ),
            # CUDAExtension(...),
      ],
      cmdclass={'build_ext': BuildExtension}
      )
