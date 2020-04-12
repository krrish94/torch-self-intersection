#include <vector>

#include <torch/torch.h>


int selfintersection_cuda_forward(at::Tensor triangles, at::Tensor selfintersections);


int selfintersection_cuda_backward(at::Tensor triangles, at::Tensor grad_triangles);


int selfintersections_forward(at::Tensor triangles, at::Tensor selfintersections) {
    return selfintersection_cuda_forward(triangles, selfintersections);
}


int selfintersections_backward(at::Tensor triangles, at::Tensor grad_triangles) {
    return selfintersection_cuda_backward(triangles, grad_triangles);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selfintersections_forward, "selfintersections forward (CUDA)");
    m.def("backward", &selfintersections_backward, "selfintersections backward (CUDA)");
}
