# This is an implementation of the Cocone algorithm
import numpy as np
from pyhull.convex_hull import ConvexHull
from math import sqrt, fabs, pi

# Read the input points and correct the dimension of the array
input_points = np.reshape(np.fromfile("bunny.xyz", sep=" "), (-1, 3))

# Split data into separate arrays for coordinates
X, Y, Z = np.transpose(input_points)

# Compute the fourth coordinate for 3D-into-4D lifting of the points to a paraboloid
W = X ** 2 + Y ** 2 + Z ** 2

# Compute a convex hull
ch = ConvexHull(np.transpose([X, Y, Z, W]).tolist())

# Filter simplices corresponding to Delaunay triangulaiton
ones4D = np.ones(4)
simplices = [s for s in ch.vertices if np.linalg.det([X[s], Y[s], Z[s], ones4D]) < 0]

# A class to store the pole information
class Pole:
    """Represents a pole with a radius and center coordinates."""
    __slots__ = ("r", "c")

    def __init__(self, r: float, c: np.ndarray):
        self.r = r
        self.c = c

class Triangle:
    """Represents a triangle as a face of a simplex, storing circumcenter information.

    Attributes:
        center_s0 (np.ndarray): The circumcenter of the first simplex defining this triangle.
        incident_vertex_s0 (int): The index of the vertex incident to the first simplex
                                  and opposite this triangle.
        center_s1 (np.ndarray | None): The circumcenter of the second simplex defining
                                      this triangle, if it exists.
    """
    __slots__ = ("center_s0", "center_s1", "incident_vertex_s0")

    def __init__(self, center_s0: np.ndarray, incident_vertex_s0: int, center_s1: np.ndarray | None = None):
        self.center_s0 = center_s0
        self.incident_vertex_s0 = incident_vertex_s0
        self.center_s1 = center_s1

    def add_second_simplex_center(self, center_s1: np.ndarray):
        """Adds the circumcenter of a second simplex that shares this triangle."""
        assert self.center_s1 is None, "Second simplex center already added."
        self.center_s1 = center_s1


# Calculates the center and radius of the circumscribed sphere for four points.
# A tuple containing the center and radius of the circumscribed sphere.
# If no valid sphere, returns ([0,0,0], -1)
def circumscribed_sphere(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, W: np.ndarray):
    A = 2.0 * np.column_stack((
        X[1:] - X[0],
        Y[1:] - Y[0],
        Z[1:] - Z[0],
    ))
    b = W[1:] - W[0]
    detA = np.linalg.det(A)
    if abs(detA) < 1e-12:
        return np.zeros(3), -1.0

    try:
        C = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros(3), -1.0
    
    r = np.linalg.norm(C - np.array([X[0], Y[0], Z[0]], dtype=float))

    if not np.isfinite(r) or r <= 0.0:
        return np.zeros(3), -1.0

    return C, float(r)

# Compute poles for each vertex
poles = {}
triangles = {}
for s in simplices:
    # Compute circumscribed sphere for each vertex
    C, r = circumscribed_sphere(X[s], Y[s], Z[s], W[s])
    for v_idx in s:
        if v_idx not in poles or poles[v_idx].r < r:
            poles[v_idx] = Pole(r, C)

    # s = [v0, v1, v2, v3] are indices of points in the current simplex
    # a, b, c are the vertices forming the triangle
    # d is the vertex opposite to the triangle in the simplex
    for a, b, c, d in (0, 1, 2, 3), (0, 2, 3, 1), (0, 1, 3, 2), (1, 2, 3, 0):
        # Get sorted vertex indices for the triangle to ensure unique key
        t_idx = tuple(sorted((s[a], s[b], s[c])))

        # If this triangle has already been encountered from another simplex
        if t_idx in triangles:
            # Add the current simplex's circumcenter as the second one
            triangles[t_idx].add_second_simplex_center(C)
        else:
            # Create a new Triangle object, storing the current simplex's circumcenter
            # and the index of the vertex opposite this triangle in the simplex (s[d])
            triangles[t_idx] = Triangle(C, s[d])

# Compute estimation for normals
##############################################################
num_points = len(X)

# Initialize normals with zeros
Nx = np.zeros(num_points, dtype=float)
Ny = np.zeros(num_points, dtype=float)
Nz = np.zeros(num_points, dtype=float)

for v_idx, pole in poles.items():
    # position of vertex v
    v = np.array([X[v_idx], Y[v_idx], Z[v_idx]], dtype=float)
    # direction from v to its pole
    dir_vec = np.asarray(pole.c, dtype=float) - v

    norm = np.linalg.norm(dir_vec)
    if norm > 0.0:
        dir_vec /= norm  # normalize to unit vector

    Nx[v_idx] = dir_vec[0]
    Ny[v_idx] = dir_vec[1]
    Nz[v_idx] = dir_vec[2]
##############################################################

# Checks if a given triangle is "good" with respect to a specific vertex and its normal
# Implements cocone intersection test
def intersection_check(triangle: Triangle, vertex_idx: int, triangle_vertices_indices: tuple) -> bool:
    # vertex v
    v = np.array([X[vertex_idx], Y[vertex_idx], Z[vertex_idx]])

    # estimated normal at v
    vn = np.array([Nx[vertex_idx], Ny[vertex_idx], Nz[vertex_idx]])

    v0_center = triangle.center_s0

    vec_v_to_v0 = v0_center - v
    norm_v_to_v0 = np.linalg.norm(vec_v_to_v0)

    # Handle division by zero for norm if v0_center and v are the same point
    if norm_v_to_v0 == 0:
        d0 = 0.0  # Points are coincident, assume it passes some checks depending on context
    else:
        d0 = np.dot(vn, vec_v_to_v0 / norm_v_to_v0)

    # cocone angle
    theta = pi / 8

    # triangle belongs to only one simplex
    if triangle.center_s1 is None:
        idx = list(triangle_vertices_indices)
        p0, p1, p2 = np.transpose([X[idx], Y[idx], Z[idx]])
        vp1, vp2 = p1 - p0, p2 - p0

        # 'ov' is the incident_vertex for the first simplex (s[d])
        ov_idx = triangle.incident_vertex_s0
        ov = np.array([t[ov_idx] for t in (X, Y, Z)])

        pr0 = np.linalg.det([vp1, vp2, v0_center - p0])
        pr1 = np.linalg.det([vp1, vp2, ov - p0])

        if pr0 * pr1 >= 0:
            return True

        # cocone intersection test
        return -theta < d0 < theta

    v1_center = triangle.center_s1

    vec_v_to_v1 = v1_center - v
    norm_v_to_v1 = np.linalg.norm(vec_v_to_v1)

    # Handle division by zero for norm
    if norm_v_to_v1 == 0:
        d1 = 0.0
    else:
        d1 = np.dot(vn, vec_v_to_v1 / norm_v_to_v1)

    d0, d1 = sorted((d0, d1))

    # cocone intersection test
    if d1 <= -theta or d0 >= theta:
        return False

    return True

# Compute the triangles that passes the test for all vertices
#################################################
candidate_triangles = {}

for t_idx, trg in triangles.items():
    # t_idx: tuple of 3 vertex indices of the triangle
    # trg: Triangle object with circumcenter info

    keep = True
    for v_idx in t_idx:
        # if triangle does not pass the intersection test at this vertex,
        # discard it
        if not intersection_check(trg, v_idx, t_idx):
            keep = False
            break

    if keep:
        candidate_triangles[t_idx] = trg
#################################################

# Output the data in OFF format to
with open("test.off", "w") as out:
    out.write("OFF\n")
    out.write("{} {} 0\n".format(len(X), len(candidate_triangles)))

    # Output the vertices
    for v in zip(X, Y, Z):
        out.write("{} {} {}\n".format(*v))

    # Output triangles
    for t_idx, trg in candidate_triangles.items():
        idx = list(t_idx)
        p0, p1, p2 = np.transpose([X[idx], Y[idx], Z[idx]])

        # Try to guess the orientation
        if trg.center_s1 is None:
            ov_idx = trg.incident_vertex_s0
            ov = np.array([t[ov_idx] for t in (X, Y, Z)])
            if np.linalg.det([p1 - p0, p2 - p0, ov - p0]) > 0:
                idx = reversed(idx)
        else:
            n = [sum(Nx[idx]), sum(Ny[idx]), sum(Nz[idx])]
            if np.linalg.det([p1 - p0, p2 - p0, n]) < 0:
                idx = reversed(idx)

        out.write("3 {} {} {}\n".format(*idx))