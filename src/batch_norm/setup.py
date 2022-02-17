from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension  # , CUDAExtension

setup(
    name='batch_norm_cpp',
    ext_modules=[
        CppExtension(
            name='batch_norm_cpu',
            sources=['batch_norm_binding.cpp'],
        ),
        # CUDAExtension(...),
    ],
    cmdclass={'build_ext': BuildExtension}
)
