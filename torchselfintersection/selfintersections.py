import torch


class SelfIntersections(torch.nn.Module):

    def __init__(self):
        super(SelfIntersections, self).__init__()

    @staticmethod
    def check_min_max(p1, q1, r1, p2, q2, r2):
        v1 = p2 - q1
        v2 = p1 - q1
        n1 = torch.cross(v1, v2)
        v1 = q2 - q1
        if torch.dot(v1, n1) > 0.:
            return 0
        v1 = p2 - p1
        v2 = r1 - p1
        n1 = torch.cross(v1, v2)
        v1 = r2 - p1
        if torch.dot(v1, n1) > 0.:
            return 0
        return 1

    @staticmethod
    def orient2d(p, q, r):
        return (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0])

    @staticmethod
    def intersection_test_vertex(p1, q1, r1, p2, q2, r2):
        if SelfIntersections.orient2d(r2, p2, q1) >= 0.:
            if SelfIntersections.orient2d(r2, q2, q1) <= 0.:
                if SelfIntersections.orient2d(p1, p2, q1) > 0:
                    if SelfIntersections.orient2d(p1, q2, q1) <= 0:
                        return 1
                    else:
                        return 0
                else:
                    if SelfIntersections.orient2d(p1, p2, r1) >= 0.:
                        if SelfIntersections.orient2d(q1, r1, p2) >= 0.:
                            return 1
                        else:
                            return 0
                    else:
                        return 0
            else:
                if SelfIntersections.orient2d(p1, q2, q1) <= 0.:
                    if SelfIntersections.orient2d(r2, q2, r1) <= 0.:
                        if SelfIntersections.orient2d(q1, r1, r2) >= 0.:
                            return 1
                        else:
                            return 0
                    else:
                        return 0
                else:
                    return 0
        else:
            if SelfIntersections.orient2d(r2, p2, r1) >= 0.:
                if SelfIntersections.orient2d(q1, r1, r2) >= 0.:
                    if SelfIntersections.orient2d(p1, p2, r1) >= 0.:
                        return 1
                    else:
                        return 0
                else:
                    if SelfIntersections.orient2d(q1, r1, q2) >= 0.:
                        if SelfIntersections.orient2d(r2, r1, q2) >= 0.:
                            return 1
                        else:
                            return 0
                    else:
                        return 0
            else:
                return 0

    @staticmethod
    def intersection_test_edge(p1, q1, r1, p2, q2, r2):
        if SelfIntersections.orient2d(r2, p2, q1) >= 0.:
            if SelfIntersections.orient2d(p1, p2, q1) >= 0.:
                if SelfIntersections.orient2d(p1, q1, r2) >= 0.:
                    return 1
                else:
                    return 0
            else:
                if SelfIntersections.orient2d(q1, r1, p2) >= 0.:
                    if SelfIntersections.orient2d(r1, p1, p2) >= 0.:
                        return 1
                    else:
                        return 0
                else:
                    return 0
        else:
            if SelfIntersections.orient2d(r2, p2, r1) >= 0.:
                if SelfIntersections.orient2d(p1, p2, r1) >= 0.:
                    if SelfIntersections.orient2d(p1, r1, r2) >= 0.:
                        return 1
                    else:
                        if SelfIntersections.orient2d(q1, r1, r2) >= 0.:
                            return 1
                        else:
                            return 0
                else:
                    return 0
            else:
                return 0

    @staticmethod
    def ccw_triangle_triangle_intersection_2d(p1, q1, r1, p2, q2, r2):
        if SelfIntersections.orient2d(p2, q2, p1) >= 0.:
            if SelfIntersections.orient2d(q2, r2, p1) >= 0.:
                if SelfIntersections.orient2d(r2, p2, p1) >= 0.:
                    return 1
                else:
                    return SelfIntersections.intersection_test_edge(p1, q1, r1, p2, q2, r2)
            else:
                if SelfIntersections.orient2d(r2, p2, p1) >= 0.:
                    return SelfIntersections.intersection_test_edge(p1, q1, r1, r2, p2, q2)
                else:
                    return SelfIntersections.intersection_test_vertex(p1, q1, r1, p2, q2, r2)
        else:
            if SelfIntersections.orient2d(q2, r2, p1) >= 0.:
                if SelfIntersections.orient2d(r2, p2, p1) >= 0.:
                    return SelfIntersections.intersection_test_edge(p1, q1, r1, q2, r2, p2)
                else:
                    return SelfIntersections.intersection_test_vertex(p1, q1, r1, q2, r2, p2)
            else:
                return SelfIntersections.intersection_test_vertex(p1, q1, r1, r2, p2, q2)

    @staticmethod
    def triangle_triangle_2d_test(p1, q1, r1, p2, q2, r2):
        if SelfIntersections.orient2d(p1, q1, r1) < 0.:
            if SelfIntersections.orient2d(p2, q2, r2) < 0.:
                return SelfIntersections.ccw_triangle_triangle_intersection_2d(p1, r1, q1, p2, r2, q2)
            else:
                return SelfIntersections.ccw_triangle_triangle_intersection_2d(p1, r1, q1, p2, q2, r2)
        else:
            if SelfIntersections.orient2d(p2, q2, r2) < 0.:
                return SelfIntersections.ccw_triangle_triangle_intersection_2d(p1, q1, r1, p2, r2, q2)
            else:
                return SelfIntersections.ccw_triangle_triangle_intersection_2d(p1, q1, r1, p2, q2, r2)

    @staticmethod
    def coplanar_triangle_triangle_3d(p1, q1, r1, p2, q2, r2, n1, n2):
        
        P1 = torch.zeros(2).to(p1)
        Q1 = P1.clone()
        R1 = P1.clone()
        P2 = P1.clone()
        Q2 = P1.clone()
        R2 = P1.clone()

        n = torch.where(n1 < 0, -n1, n1)

        # Project triangles onto 2D such that the area of projection
        # is maximized
        if n[0] > n[2] and n[0] >= n[1]:
            # Project onto YZ-plane
            P1[0], P1[1] = q1[2], q1[1]
            Q1[0], Q1[1] = p1[2], p1[1]
            R1[0], R1[1] = r1[2], r1[1]

            P2[0], P2[1] = q2[2], q2[1]
            Q2[0], Q2[1] = p2[2], p2[1]
            R2[0], R2[1] = r2[2], r2[1]

        elif n[1] > n[2] and n[1] >= n[0]:
            # Project onto XZ-plane
            P1[0], P1[1] = q1[0], q1[2]
            Q1[0], Q1[1] = p1[0], p1[2]
            R1[0], R1[1] = r1[0], r1[2]

            P2[0], P2[1] = q2[0], q2[2]
            Q2[0], Q2[1] = p2[0], p2[2]
            R2[0], R2[1] = r2[0], r2[2]

        else:
            # Project onto XY-plane
            P1[0], P1[1] = q1[0], q1[1]
            Q1[0], Q1[1] = p1[0], p1[1]
            R1[0], R1[1] = r1[0], r1[1]

            P2[0], P2[1] = q2[0], q2[1]
            Q2[0], Q2[1] = p2[0], p2[1]
            R2[0], R2[1] = r2[0], r2[1]

        return SelfIntersections.triangle_triangle_2d_test(P1, Q1, R1, P2, Q2, R2)

    @staticmethod
    def triangle_triangle_3d_test(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2, n1, n2):
        if dp2 > 0.:
            if dq2 > 0.:
                return SelfIntersections.check_min_max(p1, r1, q1, r2, p2, q2)
            elif dr2 > 0.:
                return SelfIntersections.check_min_max(p1, r1, q1, q2, r2, p2)
            else:
                return SelfIntersections.check_min_max(p1, q1, r1, p2, q2, r2)
        elif dp2 < 0.:
            if dq2 < 0.:
                return SelfIntersections.check_min_max(p1, q1, r1, r2, p2, q2)
            elif dr2 < 0.:
                return SelfIntersections.check_min_max(p1, q1, r1, r2, p2, q2)
            else:
                return SelfIntersections.check_min_max(p1, r1, q1, p2, q2, r2)
        else:
            if dq2 < 0.:
                if dr2 > 0:
                    return SelfIntersections.check_min_max(p1, r1, q2, p2, q2, r2)
                else:
                    return SelfIntersections.check_min_max(p1, q1, r1, p2, q2, r2)
            elif dq2 > 0.:
                if dr2 > 0.:
                    return SelfIntersections.check_min_max(p1, r1, q1, p2, q2, r2)
                else:
                    return SelfIntersections.check_min_max(p1, q1, r1, q2, r2, p2)
            else:
                if dr2 > 0.:
                    return SelfIntersections.check_min_max(p1, q1, r1, r2, p2, q2)
                elif dr2 < 0.:
                    return SelfIntersections.check_min_max(p1, r1, q2, r2, p2, q2)
                else:
                    return SelfIntersections.coplanar_triangle_triangle_3d(
                        p1, q1, r1, p2, q2, r2, n1, n2
                    )

    def forward(self, triangles):
        r"""Computes self-intersections in a triangle-mesh.

        Args:
            triangles: coordinates of triangles in the mesh
        """
        selfintersections = torch.zeros(triangles.shape[0], triangles.shape[0]).to(triangles)

        triangles2 = triangles.clone()

        for i in range(triangles.shape[0]):
            
            # Vertices of triangle 'i'
            p1 = triangles[i, :3]
            q1 = triangles[i, 3:6]
            r1 = triangles[i, 6:]
            
            for j in range(i + 1, triangles2.shape[0]):

                # Vertices of triangle 'j'
                p2 = triangles2[j, :3]
                q2 = triangles2[j, 3:6]
                r2 = triangles2[j, 6:]

                # Get plane of triangle 'j'
                n2 = torch.cross(p2 - r2, q2 - r2)

                # Get the dot products of p1 - r2, q1 - r2, r1 - r2, wrt n2
                # i.e., if all these vectors joining vertices of p1, q1, r1
                # to r2 are on the same side of n2, then we can safely skip
                # further self-intersection tests for this pair
                dp1 = torch.dot(p1 - r2, n2)
                dq1 = torch.dot(q1 - r2, n2)
                dr1 = torch.dot(r1 - r2, n2)
                
                if dp1 * dq1 > 0. and dp1 * dr1 > 0.:
                    continue

                # Compute the signs of distance of p2, q2, and r2 to the plane
                # of triangle (p1, q1, r1)
                
                # Plane of triangle 'i'
                n1 = torch.cross(q1 - p1, r1 - p1)

                # Get signs
                dp2 = torch.dot(p2 - r1, n1)
                dq2 = torch.dot(q2 - r1, n1)
                dr2 = torch.dot(r2 - r1, n1)

                if dp2 * dq2 > 0. and dp2 * dr2 > 0.:
                    continue

                # Now, "serious" tests for overlaps (in 3D)

                # Test permutations of triangle vertices also.

                if dp1 > 0.:
                    if dq1 > 0.:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            r1, p1, q1, p2, r2, q2, dp2, dr2, dq2, n1, n2
                        )
                    elif dr1 > 0.:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            q1, r1, p1, p2, r2, q2, dp2, dr2, dq2, n1, n2
                        )
                    else:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            p1, q1, r1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                        )
                elif dp1 < 0.:
                    if dq1 < 0.:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            r1, p1, q1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                        )
                    elif dr1 < 0.:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            q1, r1, p1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                        )
                    else:
                        selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                            p1, q1, r1, p2, r2, q2, dp2, dr2, dq2, n1, n2
                        )
                else:
                    if dq1 < 0.:
                        if dr1 >= 0.:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                q1, r1, p1, p2, r2, q2, dp2, dr2, dq2, n1, n2
                            )
                        else:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                p1, q1, r1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                            )
                    elif dq1 > 0.:
                        if dr1 >= 0:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                p1, q1, r1, p2, r2, q2, dp2, dr2, dq2
                            )
                        else:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                q1, r1, p1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                            )
                    else:
                        if dr1 > 0.:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                r1, p1, q1, p2, q2, r2, dp2, dq2, dr2, n1, n2
                            )
                        elif dr1 < 0.:
                            selfintersections[i][j] = SelfIntersections.triangle_triangle_3d_test(
                                r1, p1, q1, p2, r2, q2, dp2, dr2, dq2, n1, n2
                            )
                        else:
                            selfintersections[i][j] = SelfIntersections.coplanar_triangle_triangle_3d(
                                p1, q1, r1, p2, q2, r2, n1, n2
                            )

        return selfintersections


if __name__ == "__main__":

    selfintersector = SelfIntersections()
    triangles = torch.rand(40, 9)
    selfintersections = selfintersector(triangles)
    print(selfintersections)
