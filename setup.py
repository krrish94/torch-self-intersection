from setuptools import setup
from torch.utils import cpp_extension

package_name = "torchselfintersection"
version = "0.1.0"
requirements = [
    "torch>1.1.0",
]
long_description = "A pytorch module to compute self intersections within"

setup(
    name="torchselfintersection",
    version=version,
    description="Pytorch Self-Intersection",
    long_description=long_description,
    requirements=requirements,
    ext_modules=[
        cpp_extension.CppExtension(
            name="torchselfintersection.selfintersection_cpu",
            sources=[
                "torchselfintersection/selfintersection_cpu.cpp",
            ],
            include_dirs=cpp_extension.include_paths(),
            language='c++',
        ),
        # CUDAExtension(
        #     "selfintersection.cuda",
        #     [
        #         "torchselfintersection/selfintersection_cuda_wrapper.cpp",
        #         "torchselfintersection/selfintersection.cu",
        #     ],
        # ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
