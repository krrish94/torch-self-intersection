import torch

from torchselfintersection import selfintersection_cpu


class SelfIntersectionsCPUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, triangles):
        triangles = triangles.contiguous()
        selfintersections = selfintersection_cpu.forward(triangles)
        # The CPU function only computes self intersections for the
        # "upper triangle". Compute them for the "lower triangle" too.
        # Also, set the diagonals (trivially) to 1, cause a triangle
        # always self-intersects, itself.
        selfintersections = selfintersections + selfintersections.transpose(0, 1)
        selfintersections = selfintersections + torch.eye(selfintersections.shape[0]).to(selfintersections)
        return selfintersections

    @staticmethod
    def backward(ctx, gradtriangles):
        return None


class SelfIntersectionsCPU(torch.nn.Module):

    def __init__(self):
        super(SelfIntersectionsCPU, self).__init__()

    def forward(self, triangles):
        return SelfIntersectionsCPUFunction.apply(triangles)
    

if __name__ == "__main__":

    selfintersector = SelfIntersectionsCPU()
    triangles = torch.rand(40, 9)
    selfintersections = selfintersector(triangles)
    print(selfintersections)
