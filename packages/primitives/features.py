from __future__ import annotations

import math
from dataclasses import dataclass

from packages.common.geometry import aspect_ratio, orientation_sincos
from packages.ingest.models import Primitive
from packages.primitives.spatial_index import PrimitiveIndexes

ENTITY_FEATURE_SCHEMA_VERSION = "entity_features_v1"


@dataclass(frozen=True)
class EntityFeatures:
    primitive_id: str
    values: dict[str, float]
    schema_version: str = ENTITY_FEATURE_SCHEMA_VERSION


def extract_entity_features(
    primitive: Primitive,
    by_id: dict[str, Primitive],
    indexes: PrimitiveIndexes,
    neighbor_radius_m: float = 1.0,
) -> EntityFeatures:
    geom = primitive.geometry
    minx, miny, maxx, maxy = primitive.bbox.as_tuple()
    width = max(maxx - minx, 0.0)
    height = max(maxy - miny, 0.0)
    area = float(geom.area) if geom.geom_type in {"Polygon", "MultiPolygon"} else 0.0
    sin_a, cos_a = orientation_sincos(geom)
    endpoint_distance = primitive.length
    if geom.geom_type == "LineString" and len(geom.coords) >= 2:
        x0, y0 = geom.coords[0][0], geom.coords[0][1]
        x1, y1 = geom.coords[-1][0], geom.coords[-1][1]
        endpoint_distance = math.hypot(x1 - x0, y1 - y0)
    curvature = 0.0
    if primitive.length > 1e-9:
        curvature = abs(primitive.length - endpoint_distance) / primitive.length

    search = geom.buffer(neighbor_radius_m)
    neighbor_ids = [i for i in indexes.bbox.query(search) if i != primitive.id]
    nearby_parallel = 0
    nearby_collinear = 0
    for nid in neighbor_ids:
        other = by_id.get(nid)
        if other is None:
            continue
        osin, ocos = orientation_sincos(other.geometry)
        dot = abs(cos_a * ocos + sin_a * osin)
        if dot > 0.95:
            nearby_parallel += 1
            if primitive.geometry.distance(other.geometry) < 0.05:
                nearby_collinear += 1
    endpoint_hits = indexes.endpoint.query(geom.buffer(0.05))
    endpoint_degree = max(len(set(endpoint_hits)) - 1, 0)
    local_density = float(len(neighbor_ids))

    values = {
        "length": float(primitive.length),
        "area": area,
        "bbox_width": width,
        "bbox_height": height,
        "orientation_sin": float(sin_a),
        "orientation_cos": float(cos_a),
        "aspect_ratio": aspect_ratio(minx, miny, maxx, maxy),
        "endpoint_distance": float(endpoint_distance),
        "curvature": float(curvature),
        "layer_id": float(abs(hash(primitive.layer)) % 10_000),
        "color": float(primitive.color or 0),
        "linetype_id": float(abs(hash(primitive.linetype or "")) % 10_000),
        "source_kind_id": float(abs(hash(primitive.source_entity_type)) % 10_000),
        "block_indicator": 1.0 if primitive.block_name else 0.0,
        "neighbor_count": float(len(neighbor_ids)),
        "nearby_parallel_count": float(nearby_parallel),
        "nearby_collinear_count": float(nearby_collinear),
        "endpoint_degree": float(endpoint_degree),
        "local_density": local_density,
    }
    return EntityFeatures(primitive_id=primitive.id, values=values)
