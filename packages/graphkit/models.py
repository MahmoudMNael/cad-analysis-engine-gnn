from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from packages.contracts.enums import PortalKind


@dataclass(frozen=True)
class GraphNode:
    id: str
    point: tuple[float, float]
    incident_edge_ids: tuple[str, ...]
    source_primitive_ids: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    id: str
    node_a: str
    node_b: str
    geometry: LineString
    primitive_ids: tuple[str, ...]
    class_prior: str | None
    predicted_class: str | None
    confidence: float


@dataclass(frozen=True)
class Portal:
    id: str
    geometry: BaseGeometry
    kind: PortalKind
    source_edge_ids: tuple[str, ...]
    confidence: float
    source_primitive_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    portals: tuple[Portal, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def node_map(self) -> dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}

    def edge_map(self) -> dict[str, GraphEdge]:
        return {e.id: e for e in self.edges}

    def dangling_node_ids(self) -> list[str]:
        return [n.id for n in self.nodes if len(n.incident_edge_ids) <= 1]


@dataclass(frozen=True)
class GapCandidate:
    pair_id: str
    node_a: str
    node_b: str
    distance_m: float
    geometry: LineString
    collinear: bool
    crosses: bool
    already_connected: bool
    opening_like: bool
