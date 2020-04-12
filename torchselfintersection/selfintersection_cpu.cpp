#include <torch/extension.h>


void selfintersections_cpu_wrapper(at::Tensor triangles, at::Tensor selfintersections) {

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selfintersections_cpu", &selfintersections_cpu_wrapper, "selfintersections (CPU)");
}
