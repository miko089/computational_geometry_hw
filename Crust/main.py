import sys
from typing import List, Tuple, Set

import numpy as np
from scipy.spatial import ConvexHull


def read_points(path: str) -> np.ndarray:
    """Read 2D points from file: each line 'x y'."""
    xs = []
    ys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            x = float(parts[0])
            y = float(parts[1])
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    return pts


def lift_to_paraboloid(points_2d: np.ndarray) -> np.ndarray:
    """
    Lift 2D points (x, y) to 3D paraboloid (x, y, x^2 + y^2).
    This allows us to get 2D Delaunay triangulation from the lower hull.
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    z = x * x + y * y
    return np.column_stack([x, y, z])


def delaunay_triangles_via_convex_hull(points_2d: np.ndarray) -> np.ndarray:
    """
    Compute 2D Delaunay triangulation of points_2d using
    3D convex hull on the lifted paraboloid.
    """
    if len(points_2d) < 3:
        return np.empty((0, 3), dtype=int)

    pts3 = lift_to_paraboloid(points_2d)
    hull = ConvexHull(pts3)

    triangles = []
    # hull.simplices: facets (triangles here) as indices into pts3
    # hull.equations: corresponding plane equations a*x + b*y + c*z + d = 0
    # For lifted paraboloid, facets with c < 0 form the lower hull → Delaunay.
    for simplex, eq in zip(hull.simplices, hull.equations):
        _a, _b, c, _d = eq
        if c < 0:  # lower hull
            # simplex is a triangle (3 indices)
            triangles.append(tuple(simplex))

    if not triangles:
        return np.empty((0, 3), dtype=int)

    return np.array(triangles, dtype=int)


def circumcenter_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute circumcenter of triangle ABC in 2D.
    """
    ba = b - a
    ca = c - a

    M = np.array([[ba[0], ba[1]],
                  [ca[0], ca[1]]], dtype=float)

    rhs = np.array([
        0.5 * (np.dot(b, b) - np.dot(a, a)),
        0.5 * (np.dot(c, c) - np.dot(a, a)),
    ], dtype=float)

    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        # Very flat / degenerate triangle: approximate by average
        return (a + b + c) / 3.0

    u = np.linalg.solve(M, rhs)
    return u


def compute_voronoi_vertices_from_delaunay(points_2d: np.ndarray,
                                           triangles: np.ndarray) -> np.ndarray:
    """
    Given 2D points and their Delaunay triangles, compute Voronoi vertices
    as circumcenters of the triangles.
    """
    centers = []
    for tri in triangles:
        i, j, k = tri
        a = points_2d[i]
        b = points_2d[j]
        c = points_2d[k]
        center = circumcenter_2d(a, b, c)
        centers.append(center)
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.vstack(centers)


def crust_edges(points_2d: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Compute crust edges for 2D sample points using Voronoi filtering
    and convex hull based Delaunay.
    """
    n = len(points_2d)
    if n < 2:
        return set()

    # Step 1: Delaunay of original points → Voronoi vertices
    tri_p = delaunay_triangles_via_convex_hull(points_2d)
    vor_vertices = compute_voronoi_vertices_from_delaunay(points_2d, tri_p)

    # Step 2: Q = P ∪ V
    if len(vor_vertices) == 0:
        q_points = points_2d.copy()
    else:
        q_points = np.vstack([points_2d, vor_vertices])

    # Step 3: Delaunay of Q via convex hull
    tri_q = delaunay_triangles_via_convex_hull(q_points)

    # Step 4: collect edges whose endpoints are both in original points
    edges: Set[Tuple[int, int]] = set()

    for tri in tri_q:
        i, j, k = tri
        # All triples of edges in triangle
        for u, v in ((i, j), (j, k), (k, i)):
            # Keep only edges whose endpoints are within [0, n)
            if u < n and v < n:
                if u == v:
                    continue
                # Store edges with sorted endpoints to avoid duplicates
                a = min(u, v)
                b = max(u, v)
                edges.add((a, b))

    return edges

def plot_crust(points_2d: np.ndarray, edges: Set[Tuple[int, int]]):
    import matplotlib.pyplot as plt

    xs = points_2d[:, 0]
    ys = points_2d[:, 1]

    _fig, ax = plt.subplots(figsize=(6, 6))

    # points aren't too visible under edges, but we all know that there is nup under
    # edges so I didn't fix it :)
    ax.scatter(xs, ys, s=5, c="0.7", edgecolors="none", zorder=1)

    for i, j in edges:
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[j]
        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=3.0,
            color="tab:red",
            zorder=2,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Crust reconstruction")
    plt.tight_layout()
    plt.show()

def main(argv: List[str]) -> None:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} points.txt [--plot]", file=sys.stderr)
        sys.exit(1)

    points_path = argv[1]
    do_plot = "--plot" in argv

    pts = read_points(points_path)
    edges = crust_edges(pts)

    for i, j in sorted(edges):
        print(i, j)

    if do_plot:
        plot_crust(pts, edges)


if __name__ == "__main__":
    main(sys.argv)
