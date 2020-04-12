from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CPPExtension, CUDAExtension

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
        CPPExtension(
            "selfintersection.cpu",
            [
                "torchselfintersection/selfintersection_cpu.cpp",
            ],
        ),
        CUDAExtension(
            "selfintersection.cuda",
            [
                "torchselfintersection/selfintersection_cuda_wrapper.cpp",
                "torchselfintersection/selfintersection.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
