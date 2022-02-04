from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension  # , CUDAExtension

setup(
    name='conv_cpp',
    ext_modules=[
        CppExtension(
            name='conv_cpu',
            sources=['conv_binding.cpp'],
        ),
        # CUDAExtension(...),
    ],
    cmdclass={'build_ext': BuildExtension}
)
