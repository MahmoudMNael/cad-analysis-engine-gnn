"""Bounded endpoint snapping for compatible wall endpoints only."""

from __future__ import annotations

from collections import defaultdict

from packages.common.configuration import Settings
from packages.common.geometry import round_point


def _cell(point: tuple[float, float], cell: float) -> tuple[int, int]:
    return (int(point[0] // cell), int(point[1] // cell))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def snap_endpoints(
    points: list[tuple[float, float]],
    settings: Settings,
) -> list[tuple[float, float]]:
    """Cluster nearby endpoints and replace each with the rounded cluster centroid."""
    if not points:
        return []
    n = len(points)
    uf = UnionFind(n)
    cell = max(settings.snap_tolerance_m, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, pt in enumerate(points):
        buckets[_cell(pt, cell)].append(i)
    tol2 = settings.snap_tolerance_m ** 2
    for i, pt in enumerate(points):
        cx, cy = _cell(pt, cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    if j <= i:
                        continue
                    ox, oy = points[j]
                    if (pt[0] - ox) ** 2 + (pt[1] - oy) ** 2 <= tol2:
                        uf.union(i, j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    mapping = [points[i] for i in range(n)]
    precision = settings.coord_precision
    for members in clusters.values():
        sx = sum(points[i][0] for i in members) / len(members)
        sy = sum(points[i][1] for i in members) / len(members)
        snapped = round_point((sx, sy), precision)
        for i in members:
            mapping[i] = snapped
    return mapping
