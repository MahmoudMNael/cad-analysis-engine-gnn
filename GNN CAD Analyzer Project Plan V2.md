# GNN CAD Redesign — Implementation-Grade Coding Agent Plan

## 0. Purpose

This document is the **authoritative implementation plan** for rebuilding the CAD layout engine from scratch.

The goal is not merely to reproduce the current heuristic pipeline. The goal is to produce a **modular hybrid CAD analysis platform** where:

1. deterministic computational geometry owns topology, coordinates, polygon validity, and area calculations;
2. GNNs own classification and graph-completion decisions where heuristics are unreliable;
3. every analysis produces a durable, reproducible snapshot;
4. every user correction becomes an explicit event that can become training data;
5. models are versioned and pinned per analysis job;
6. the HTTP API is strictly typed and OpenAPI-complete;
7. the system can later be split into services without rewriting domain internals.

The source design explicitly requires this hybrid approach: geometry produces legal room polygons, GNNs classify/complete the graph, and user edits are labels rather than detector state. fileciteturn0file0L27-L33

---

# 1. Non-negotiable engineering rules

The coding agent MUST follow these rules.

### 1.1 No architectural improvisation

Do not introduce a framework, package, architectural pattern, database, queue, ML library, or API convention that is not required by this document.

If an implementation detail is unspecified, prefer the smallest implementation that preserves the contracts and invariants below.

Do not redesign the architecture during implementation.

### 1.2 Python/runtime

- Python 3.12.
- FastAPI only.
- Pydantic v2.
- Shapely for computational geometry.
- ezdxf for DXF parsing.
- NetworkX for graph utilities where useful.
- PyTorch + PyTorch Geometric for GNNs.
- PostgreSQL for relational state.
- Redis for queue/job coordination.
- object storage in /object-storage directory

Python 3.14 is explicitly out of scope. fileciteturn0file0L375-L382

### 1.3 No in-memory source of truth

An in-process `Scene`, cache, or Python object MUST NEVER be the authoritative state for a completed analysis.

PostgreSQL + object storage are authoritative.

Workers may hold temporary objects during one pipeline execution.

The existing in-memory scene/job model is intentionally discarded. fileciteturn0file0L94-L106

### 1.4 No untyped API contracts

Every HTTP request body and response MUST be a Pydantic DTO.

Never return raw dictionaries from routes.

Never use `extra="allow"` for public layout contracts.

Request DTOs use:

```python
model_config = ConfigDict(extra="forbid")
```

The public API must expose the complete schema through `/openapi.json`. fileciteturn0file0L243-L254

### 1.5 Geometry and ML responsibilities must remain separate

Never make the GNN generate arbitrary polygons.

Never replace polygonization with a neural decoder.

The GNN may:

- classify CAD primitives;
- score candidate graph links;
- classify generated faces.

Geometry must:

- normalize coordinates;
- snap;
- node intersections;
- build topology;
- construct portals;
- polygonize;
- calculate areas/perimeters/bounds;
- validate polygons.

This separation is a hard architectural boundary. fileciteturn0file0L27-L33

### 1.6 Stable IDs are mandatory

Every primitive, graph node, graph edge, portal, face, room, and feedback target must have a deterministic stable identifier wherever possible.

IDs must not depend on Python object identity, list position, or random UUID generation for domain entities.

The existing design requires stable hashes so feedback can survive deterministic re-runs. fileciteturn0file0L156-L166

---

# 2. Target repository structure

Implement exactly this conceptual structure:

```text
apps/
  api/
    main.py
    app.py
    routes/
      jobs.py
      feedback.py
      layout.py
      models.py
      health.py
      admin.py
    dependencies.py
    exception_handlers.py

  worker/
    main.py
    jobs/
      analyze_job.py
    pipeline/
      runner.py

  trainer/
    main.py
    compile.py
    train.py
    evaluate.py
    promote.py

packages/
  contracts/
    dto/
      common.py
      jobs.py
      layout.py
      feedback.py
      models.py
      health.py
      errors.py
    enums.py
    schemas.py
    examples.py

  ingest/
    models.py
    dxf_loader.py
    normalization.py
    blocks.py
    units.py

  primitives/
    models.py
    ids.py
    features.py
    spatial_index.py

  graphkit/
    models.py
    snapping.py
    noding.py
    portals.py
    graph_builder.py
    candidates.py
    validation.py

  models_nn/
    common.py
    features/
      entity.py
      gap.py
      face.py
    entity_gnn.py
    gap_gnn.py
    face_gnn.py
    runtime.py
    registry.py
    thresholds.py

  assemble/
    polygonize.py
    face_features.py
    face_filters.py
    rooms.py
    openings.py
    objects.py
    texts.py
    measurements.py
    warnings.py
    layout.py
    validation.py

  views/
    models.py
    split.py
    merge.py
    delete.py
    restore.py
    reducer.py

  store/
    db/
      models/
      repositories/
      migrations/
    object_store.py
    jobs.py
    events.py
    snapshots.py
    models_registry.py
    idempotency.py

  eval/
    metrics.py
    datasets.py
    gold_cases.py
    regression.py

  common/
    errors.py
    logging.py
    hashing.py
    geometry.py
    timing.py
    configuration.py

infra/
  docker-compose.yml
  postgres/
  redis/
  migrations/
```

The important architectural boundary is:

```text
ingest      -> Document
primitives  -> PrimitiveSet
graphkit    -> PlanGraph
assemble    -> FaceSet -> Layout
models_nn   -> scores on existing stable IDs
views       -> derived Layout only
store       -> persistence
contracts   -> public wire format
```

This preserves the source document's stated package boundaries and prevents a new god-object from replacing the old `Scene`. fileciteturn0file0L61-L89

---

# 3. Core domain contracts

Create immutable/internal domain models with explicit schemas.

## 3.1 Document

```python
Document(
    document_id,
    source_hash,
    format,
    unit,
    unit_source,
    bbox,
    primitives,
    metadata,
)
```

Required invariants:

- all geometry is internally represented in metres;
- original CAD coordinates are not silently discarded;
- unit provenance is recorded;
- paperspace is excluded;
- block inserts are flattened while retaining original block name;
- every primitive has a source reference.

---

# 4. Primitive model

Supported primitive kinds initially:

```text
line
arc
polyline
lwpolyline
circle
hatch
text
block
```

Each primitive must contain:

```python
Primitive(
    id,
    kind,
    geometry,
    layer,
    color,
    linetype,
    source_entity_type,
    source_handle,
    block_name,
    attributes,
    bbox,
    length,
)
```

`source_handle` is retained for debugging/auditing when supplied by the CAD file.

The primitive ID should be deterministic from:

```text
source document hash
+ source handle when available
+ primitive kind
+ normalized geometry
+ relevant CAD attributes
```

Do not use only the source handle because some transformed/generated primitives can lose or duplicate handles.

---

# 5. Coordinate normalization

## 5.1 Required normalization sequence

For every document:

1. detect unit;
2. resolve explicit user unit if supplied;
3. otherwise use CAD unit metadata when trustworthy;
4. otherwise use the engine's configured fallback;
5. convert to metres;
6. flatten block transforms;
7. normalize geometry;
8. remove invalid/empty primitives;
9. preserve original metadata.

Never mix feet/mm/inches with internal metres.

## 5.2 Unit provenance

Store:

```text
unit = m
unit_source = explicit | cad_metadata | fallback
scale_factor
```

The layout metadata must expose the final unit.

The source design explicitly requires unit provenance. fileciteturn0file0L110-L121

---

# 6. Stable geometry IDs

Create one canonical geometry hashing function.

```python
stable_hash(
    kind,
    rounded_coordinates,
    relevant_attributes,
)
```

Rules:

- normalize coordinate ordering;
- round using one globally configured precision;
- canonicalize polygon winding before hashing;
- canonicalize unordered endpoint pairs;
- never duplicate hashing logic in multiple modules.

The same logical object must receive the same ID after deterministic reprocessing of the same document.

---

# 7. PlanGraph contract

`PlanGraph` is the canonical graph between primitive detection and polygonization.

```python
PlanGraph(
    nodes,
    edges,
    portals,
    metadata,
)
```

## Node

```text
id
point
incident_edge_ids
source_primitive_ids
attributes
```

## Edge

```text
id
node_a
node_b
geometry
primitive_ids
class_prior
predicted_class
confidence
```

## Portal

```text
id
geometry
kind
source_edge_ids
confidence
```

Portal kinds:

```text
door
window
opening
unknown
```

The existing design explicitly requires gap closure to add portal-aware topology and to prevent inventing long walls across rooms. fileciteturn0file0L114-L121

---

# 8. Pipeline orchestration contract

Every processing stage implements a common interface:

```python
class PipelineBlock(Protocol):
    name: str
    input_schema: type
    output_schema: type

    def run(self, input, context) -> output:
        ...
```

Each block MUST expose:

```text
name
version
input_schema
output_schema
implementation_sha
model_ref
timings
warnings
```

A block may have either:

```text
model_ref = "heuristic:<version>"
```

or:

```text
model_ref = "gnn:<model_name>:<version>"
```

The source plan explicitly requires replaceable blocks with schema/model references. fileciteturn0file0L110-L123

---

# 9. Exact runtime pipeline

The worker MUST execute these stages in this exact order:

```text
1. ingest
2. primitive indexing
3. entity classification
4. graph construction
5. gap candidate generation
6. gap classification
7. graph closure
8. deterministic polygonization
9. face feature extraction
10. face classification
11. room assembly
12. openings
13. objects
14. texts
15. measurements
16. warnings
17. snapshot
18. initial derived view
19. persist success
```

No stage may silently skip another stage.

Each stage writes its timing and warnings.

---

# 10. Stage 1 — Ingest

## Required behavior

Input:

- uploaded DXF/DWG-compatible source;
- or an approved R2 URL.

Output:

```text
Document
```

Rules:

- reject unsupported formats;
- never silently continue after malformed CAD parsing;
- reject impossible unit/coordinate conditions;
- preserve source file in object storage;
- compute source SHA-256;
- assign job ID before processing;
- record original file metadata.

Errors must map to explicit API error codes.

---

# 11. Stage 2 — Primitive index

Create spatial indexes immediately after primitive extraction.

Required indexes:

```text
primitive bbox index
endpoint index
text bbox index
block bbox index
```

Do not perform repeated O(N²) scans across the full document when a spatial query can limit candidates.

---

# 12. Stage 3 — GNN-A EntityClass

## Classes

```text
wall
opening
furniture
hatch
dim
text
unknown
```

Heuristic layer/block-name hints are **features**, not final decisions. fileciteturn0file0L114-L118

## Required feature groups

Each primitive feature vector should include, where applicable:

### Geometry

```text
length
area
bbox_width
bbox_height
orientation_sin
orientation_cos
aspect_ratio
endpoint_distance
curvature
```

### CAD metadata

```text
layer embedding/id
color
linetype
source entity kind
block-name indicators
```

### Local context

```text
neighbor count
nearby parallel count
nearby collinear count
endpoint degree
local density
```

Feature extraction MUST be deterministic and versioned.

---

# 13. Entity GNN output

For every primitive/edge:

```python
EntityScore(
    target_id,
    probabilities,
    predicted_class,
    confidence,
    entropy,
    model_version,
)
```

Store the complete probability vector, not just argmax.

The snapshot must preserve both heuristic and model scores because the original plan explicitly requires later analysis/distillation. fileciteturn0file0L146-L152

---

# 14. Entity heuristic fallback

Before any trained model exists, use a deterministic heuristic classifier.

It MUST return the exact same output shape as the GNN runtime.

Example:

```text
heuristic wall likelihood
heuristic opening likelihood
...
```

The pipeline must therefore never contain:

```python
if model_available:
    special_pipeline()
else:
    completely_different_pipeline()
```

Instead:

```python
entity_scores = entity_runtime.predict(graph)
```

where runtime internally chooses model or heuristic.

This ensures model replacement does not alter downstream contracts.

---

# 15. Stage 4 — Graph construction

Only primitives predicted/configured as wall candidates may contribute wall topology.

Required sequence:

```text
candidate wall primitives
    ↓
endpoint extraction
    ↓
spatial snapping
    ↓
segment intersection detection
    ↓
segment splitting
    ↓
node creation
    ↓
edge creation
    ↓
portal association
    ↓
graph validation
```

## Important

Do not snap geometry globally.

Only snap compatible nearby endpoints/intersections.

Snapping must be bounded by configured tolerances.

Never join geometrically unrelated objects merely because they are close.

---

# 16. Stage 5 — Gap candidate generation

Generate candidate links only from dangling/under-connected nodes.

Constraints:

```text
distance <= configured d_max
candidate must not already exist
candidate must pass orientation/geometric plausibility rules
candidate must not cross prohibited geometry
```

The source plan gives an approximate 1–2 m search radius and explicitly prohibits long invented walls. fileciteturn0file0L117-L119

Make the actual radius configurable.

---

# 17. Stage 6 — GNN-B GapClose

Input:

```text
candidate node pair
local graph neighborhood
geometry relationship
```

Output:

```python
GapScore(
    pair_id,
    probability,
    decision,
    confidence,
    model_version,
)
```

Training positives:

- explicit `draw_wall`;
- explicit `connect_nodes`;
- trusted synthetic gaps.

Training negatives:

- rejected candidates;
- known non-wall boundary gaps;
- sampled hard negatives.

Never train only on random negatives.

---

# 18. Gap decision rules

Use a configurable threshold:

```text
P(close) >= threshold
```

But ALSO require structural constraints:

```text
distance <= d_max
geometry valid
does not create invalid geometry
does not produce forbidden giant faces
```

The model score alone is never sufficient to modify topology.

---

# 19. Stage 7–8 — Polygonization

Use deterministic Shapely operations.

Required sequence:

```text
closed wall graph
    ↓
line merge
    ↓
noding
    ↓
polygonize
    ↓
face normalization
    ↓
face IDs
    ↓
face metadata
```

Every generated face must be persisted, including faces that will later be discarded.

This is critical because discarded faces contain the negative examples needed for indoor/outdoor/cavity learning. fileciteturn0file0L170-L185

---

# 20. Face validity checks

Every face must pass:

```text
is_valid
area > min_face_area
finite coordinates
no NaN/inf
reasonable bounds
```

Invalid geometry must produce a warning/error record rather than silently disappearing.

Repair attempts may use Shapely-compatible deterministic repair, but repairs must be recorded.

---

# 21. Face feature extraction

For every face compute:

```text
area
perimeter
compactness
centroid
bbox
aspect_ratio
number_of_doors
number_of_windows
wall_overlap
text_overlap
adjacent_face_count
adjacent_unbounded
boundary_complexity
```

The source design specifically identifies area, compactness, door count, text overlap, wall-solid overlap and adjacency to the unbounded face as face features. fileciteturn0file0L119-L121

---

# 22. GNN-C FaceClass

Classes:

```text
indoor
corridor
shaft
outdoor
cavity
```

Graph:

```text
face node
face adjacency edge
shared-boundary features
```

Output:

```python
FaceScore(
    face_id,
    probabilities,
    predicted_class,
    confidence,
    entropy,
    model_version,
)
```

This is the first model to train in production.

The source plan intentionally prioritizes GNN-C because today's indoor/cavity/envelope errors are high-value failures. fileciteturn0file0L146-L150

---

# 23. Face heuristic fallback

Until there is a promoted GNN-C model, use a deterministic classifier based on:

```text
area thresholds
boundary relationship
unbounded-face adjacency
portal count
text presence
wall density
containment/geometry
```

The heuristic MUST return the same score structure as the GNN.

Every decision must include:

```text
decision_source = heuristic | model
explanation
confidence
```

---

# 24. Room assembly

A physical room is derived only from accepted faces.

Rules:

```text
indoor -> physical room
corridor -> physical room
shaft/outdoor/cavity -> not a normal room
```

Preserve rejected faces in `FaceSet`.

For every physical room calculate:

```text
boundary
area_m2
perimeter_m
center
bbox
confidence
kind
neighbor_ids
door_ids
window_ids
object_ids
```

The source layout contract requires both `rooms` and immutable `physical_rooms`. fileciteturn0file0L156-L166

---

# 25. Physical rooms vs selectable rooms

This distinction is mandatory.

### `physical_rooms`

Output of the detector.

Never mutated by user virtual dividers.

### `rooms`

Derived user-facing view.

May reflect:

- split;
- merge;
- delete;
- restore;
- other view operations.

This means user corrections do not corrupt the detector graph.

---

# 26. Openings module

Openings are a separate assembly module.

Inputs:

```text
PlanGraph
GNN-A labels
portal candidates
opening heuristics
```

Outputs:

```text
doors
windows
```

Do not hard-code opening detection inside room polygonization.

---

# 27. Objects module

Object detection is a separate assembly module.

Input:

```text
primitive entities classified as furniture/objects
```

Assignment:

```text
object geometry
    ↓
point/geometry containment
    ↓
physical/selectable room
```

If no room contains the object:

```text
unassigned_objects
```

Never silently discard objects.

---

# 28. Text module

Text is preserved as first-class output.

Each text entity should store:

```text
id
content
position
bbox
rotation
layer
confidence
room_id?
```

Room names may be inferred from text overlap/containment, but inferred names must not destroy the original text content.

---

# 29. Measurements

Measurements are separate from wall/entity detection.

Store them with explicit source metadata.

Do not use dimensions as walls just because they are spatially near a wall.

---

# 30. Warning system

Warnings must be structured DTOs.

Example:

```python
WarningDto(
    code,
    severity,
    message,
    target_ids,
)
```

Examples:

```text
UNIT_INFERRED
LOW_ENTITY_CONFIDENCE
LOW_FACE_CONFIDENCE
GAP_CLOSE_REJECTED
INVALID_FACE_REPAIRED
OBJECT_UNASSIGNED
AMBIGUOUS_ROOM
NO_ROOMS_FOUND
CAD_PARSE_WARNING
```

Warnings are data, not free-form log strings.

---

# 31. Snapshot is mandatory

Every successful analysis creates an immutable snapshot before the API reports success.

Required artifact tree:

```text
jobs/{job_id}/
  source.dxf
  document.json
  plan_graph.json
  faces.parquet
  layout.json
  meta.json
```

The source design explicitly requires these artifacts for every analyze, including jobs with zero edits. fileciteturn0file0L170-L185

---

# 32. Snapshot metadata

`meta.json` MUST contain:

```text
job_id
source_hash
engine_sha
layout_schema
pipeline_version
feature_schema_versions
model_versions
model_decision_sources
unit
unit_source
created_at
stage_timings
warnings
configuration_hash
success
```

Also store thresholds/configuration used by the job.

A job must be reproducible from:

```text
source_hash
engine_sha
pipeline_version
model_versions
feature_schema_versions
configuration_hash
```

---

# 33. Model pinning

At job start:

```text
resolve currently promoted model versions
```

Then persist:

```text
entity_model_version
gap_model_version
face_model_version
```

All pipeline stages use those versions.

A model publish during a running job MUST NOT affect that job.

This is an explicit requirement in the source design. fileciteturn0file0L227-L239

---

# 34. Job state machine

Valid states:

```text
queued
running
succeeded
failed
```

Transitions:

```text
queued -> running
running -> succeeded
running -> failed
```

No direct:

```text
queued -> succeeded
queued -> failed
```

without worker execution.

Store:

```text
started_at
completed_at
error_code
error_detail
attempt
worker_id
```

A retry MUST reuse the same `job_id`.

---

# 35. Idempotency

A single job must produce at most one successful immutable snapshot.

Use a database transaction + unique constraints to guard:

```text
job_id
snapshot version
event sequence
```

If worker execution is repeated after a crash:

1. inspect job state;
2. inspect snapshot artifacts;
3. resume or safely mark the correct final state;
4. never create duplicate training snapshots.

The source plan explicitly requires idempotent jobs. fileciteturn0file0L337-L344

---

# 36. PostgreSQL core tables

At minimum implement:

```text
jobs
job_artifacts
feedback_events
layout_revisions
model_registry
training_runs
```

Suggested `jobs` columns:

```text
id
source_hash
status
source_uri
unit
unit_source
consent_training
engine_sha
pipeline_version
model_versions_json
configuration_hash
error_code
error_detail
created_at
started_at
completed_at
```

Suggested `feedback_events`:

```text
id
job_id
sequence
ts
actor
action
target_ids
payload_json
layout_rev
retracted_by_event_id
```

The source design requires event history rather than only last-state storage. fileciteturn0file0L185-L208

---

# 37. Event sourcing rules for feedback

Feedback is append-only.

Never overwrite the previous event.

Every event must have:

```text
event_id
job_id
monotonic sequence
timestamp
actor
action
target IDs
payload
layout revision
```

Undo is itself an event.

Do not delete old feedback from the event log.

---

# 38. Required feedback actions

Implement these from the beginning:

```text
accept
delete_room
restore_face
merge_rooms
split
draw_wall
connect_nodes
mark_entity
rename_room
undo
```

The source design explicitly states that dividers alone are insufficient training data. fileciteturn0file0L194-L208

---

# 39. Feedback semantics

## delete_room

Effect:

```text
physical room remains
derived view removes it
```

Training interpretation:

```text
face should not be indoor
```

## restore_face

Training interpretation:

```text
face should be indoor
```

## merge_rooms

Training interpretation:

```text
shared boundary should not remain a room boundary
```

## split

Training interpretation:

```text
parent remains indoor
new partition is required
```

## draw_wall / connect_nodes

Training interpretation:

```text
GNN-B positive link
```

## mark_entity

Training interpretation:

```text
GNN-A label
```

## rename_room

Training target for label/name behavior only.

## accept

Weak positive signal only.

The source design states that implicit accept/bounce should have zero training weight. fileciteturn0file0L198-L208

---

# 40. Layout view reducer

Implement one pure reducer:

```python
reduce_view(
    physical_layout,
    events
) -> LayoutDto
```

It must be deterministic.

It MUST NOT mutate:

```text
Document
PrimitiveSet
PlanGraph
FaceSet
```

It may produce:

```text
rooms
virtual_dividers
selection state
user-visible derived relationships
```

This is the core mechanism that keeps user edits separate from detector state.

---

# 41. Layout revisions

Every accepted feedback event increments:

```text
layout_rev += 1
```

Store the revision returned to the client.

A client may optionally submit:

```text
expected_layout_rev
```

to prevent conflicting edits.

If revision mismatches:

```text
409 LAYOUT_REVISION_CONFLICT
```

---

# 42. Public API

Base path:

```text
/api/v1
```

Required routes:

```text
POST   /jobs
POST   /jobs/from-url
GET    /jobs/{job_id}
GET    /jobs/{job_id}/layout

POST   /jobs/{job_id}/events
POST   /jobs/{job_id}/dividers
DELETE /jobs/{job_id}/dividers
POST   /jobs/{job_id}/accept
GET    /jobs/{job_id}/faces

GET    /models

POST   /admin/train

GET    /health
```

These match the source API design. fileciteturn0file0L287-L321

---

# 43. API response envelope

Success:

```json
{
  "success": true,
  "data": {},
  "message": "..."
}
```

Error:

```json
{
  "success": false,
  "error": {
    "code": "JOB_NOT_FOUND",
    "detail": "...",
    "fields": {}
  }
}
```

Use one consistent exception translation layer.

Do not leak raw framework exceptions.

---

# 44. API error codes

At minimum:

```text
INVALID_REQUEST
UNSUPPORTED_FORMAT
FILE_TOO_LARGE
CAD_PARSE_ERROR
CONVERSION_ERROR
JOB_NOT_FOUND
JOB_NOT_READY
JOB_FAILED
LAYOUT_NOT_FOUND
LAYOUT_REVISION_CONFLICT
INVALID_FEEDBACK_ACTION
INVALID_TARGET
EVENT_ALREADY_RETRACTED
MODEL_NOT_AVAILABLE
STORAGE_ERROR
QUEUE_ERROR
INTERNAL_ERROR
```

Map errors to documented HTTP statuses.

---

# 45. Strict Layout DTO

`LayoutDto` must contain:

```text
unit
meta
rooms
physical_rooms
walls
doors
windows
objects
unassigned_objects
texts
lights
virtual_dividers
warnings
```

`RoomDto` must contain the fields defined by the source contract:

```text
id
boundary
area_m2
area_net_m2
area_gross_m2
center
perimeter_m
bbox
confidence
name
kind
selectable
neighbor_ids
door_ids
window_ids
object_ids
physical_room_id
parent_room_id
dimensions?
```

Source: fileciteturn0file0L295-L301

---

# 46. DTO validation rules

Examples:

```text
polygon vertices >= 3
area_m2 >= 0
confidence in [0,1]
IDs non-empty
enum fields valid
bbox min <= max
coordinates finite
```

Never rely solely on TypeScript/frontend validation.

Backend must enforce all public invariants.

---

# 47. OpenAPI quality gate

CI MUST:

1. start the FastAPI app;
2. export `/openapi.json`;
3. compare against the committed OpenAPI snapshot;
4. fail on undocumented drift.

This is required to ensure the DTO layer remains the source of truth. fileciteturn0file0L243-L254

---

# 48. Storage consistency

For every completed job:

```text
Postgres says succeeded
AND
all required artifacts exist
AND
layout.json validates against LayoutDto
AND
meta.json exists
```

Do not mark success before artifact persistence completes.

For failure:

```text
job.status = failed
error_code != null
error_detail != null
```

Partial artifacts may remain for diagnostics but must never be presented as a successful layout.

---

# 49. Feedback lake

Compile:

```text
snapshot + event history
```

into:

```text
face_cls.parquet
edge_cls.parquet
link_pred.parquet
```

Never split train/validation/test by face.

Split by:

```text
job_id / building
```

This prevents leakage between faces from the same building. fileciteturn0file0L210-L220

---

# 50. Training row schema

Each dataset row MUST contain:

```text
sample_id
job_id
building_id
target_id
feature_schema_version
features
label
label_source
weight
engine_sha
model_input_version
```

Possible `label_source` values:

```text
expert
user_explicit
synthetic
weak
```

Use explicit weights.

---

# 51. Training data rules

Never treat these as equivalent:

```text
explicit expert correction
user correction
implicit acceptance
heuristic prediction
```

Training weight must be controlled by source.

Default:

```text
expert explicit       = high
user explicit         = medium/high
synthetic              = configured
weak accept            = 0
heuristic output       = not gold
```

Do not manufacture labels from detector predictions.

---

# 52. Training eligibility

Do not train GNN-C until the dataset contains enough explicit face corrections to make training meaningful.

The original plan's initial gate is approximately:

```text
50 explicitly labeled CAD files
```

from expert or user delete/restore feedback. fileciteturn0file0L368-L371

This number is a minimum operational gate, not a guarantee of quality.

The trainer must validate dataset size and diversity before training.

---

# 53. Trainer pipeline

Trainer order:

```text
compile
↓
validate dataset
↓
split by job/building
↓
train
↓
evaluate
↓
compare production metrics
↓
candidate registry entry
↓
shadow evaluation
↓
promotion
```

Never promote a checkpoint directly after training.

---

# 54. Evaluation metrics

Minimum metrics:

```text
face classification:
  macro F1
  per-class F1
  confusion matrix

layout:
  room IoU @ 0.5
  room count error
  false room/delete-rate proxy

gap:
  precision
  recall
  false-close rate
```

The source plan explicitly requires face F1, room IoU@0.5, count error and a delete-rate proxy. fileciteturn0file0L227-L233

---

# 55. Promotion rules

A candidate may only become production if:

```text
new model beats production on target metric
AND
no protected metric regresses beyond configured tolerance
AND
held-out evaluation passes
AND
artifact is valid
AND
feature schema version is compatible
```

Registry statuses:

```text
candidate
shadow
promoted
rejected
rolled_back
```

Keep previous promoted model for rollback.

---

# 56. Model registry

Required fields:

```text
model_name
version
checkpoint_uri
metrics_json
status
trained_on_engine_sha
feature_schema_version
created_at
promoted_at
```

This follows the source registry requirement. fileciteturn0file0L237-L239

---

# 57. Shadow mode

Shadow inference MUST NOT modify user-visible layouts.

During shadow:

```text
production model -> actual result
candidate model  -> parallel score
```

Record:

```text
candidate predictions
production predictions
difference
metrics
```

Only after shadow checks pass may the candidate be promoted.

---

# 58. Rollback

Rollback must be one atomic registry operation.

Never delete an older model.

A rollback changes:

```text
status/promoted pointer
```

not the model artifact.

Existing jobs keep their original model versions forever.

---

# 59. Retraining schedule

Support:

```text
weekly
OR
explicit-event-count threshold
OR
manual admin trigger
```

Initial configured threshold:

```text
>= 2000 new explicit events
```

as suggested by the source plan. fileciteturn0file0L227-L233

Make thresholds configuration-driven.

---

# 60. Testing strategy

The coding agent MUST build tests continuously.

## Unit tests

Every package must have unit tests for:

```text
ID hashing
unit conversion
geometry normalization
snapping
intersection splitting
candidate generation
polygonization
face feature extraction
heuristic classification
layout reducer
feedback event validation
DTO validation
```

## Integration tests

Test:

```text
API -> DB -> Redis -> worker -> object storage
```

with local Docker dependencies.

## Golden tests

Maintain a private/frozen set of real DXFs with expected gold JSON.

The source plan requires synthetic geometry in CI plus a frozen private real-DXF regression set. fileciteturn0file0L337-L344

---

# 61. Synthetic geometry test suite

Before real CAD testing, implement deterministic synthetic cases:

```text
single rectangle
two adjacent rooms
three rooms
L-shaped room
T-junction
door gap
window opening
small cavity
outdoor enclosure
nested cavity
crossing lines
nearly touching walls
duplicated wall
overlapping wall
slightly offset endpoints
long unrelated parallel lines
dimension lines near walls
furniture near walls
invalid/self-intersecting geometry
```

For each case assert:

```text
face count
room count
area
topology
IDs
warnings
```

---

# 62. Regression invariants

The following must never regress without an explicit version change:

```text
same input + same engine/model/config -> same stable IDs
same input + same versions -> same layout
re-running worker -> no duplicate snapshot
user split -> physical_rooms unchanged
undo -> previous derived state reconstructed
dropped faces remain persisted
failed jobs never appear successful
model publish cannot affect running jobs
```

---

# 63. Performance rules

Initial target:

```text
one CAD document per worker process
```

as required by the source design. fileciteturn0file0L337-L341

Use spatial indexes to avoid unnecessary O(N²) operations.

GNN inference may run on CPU.

Training uses GPU when available.

Do not prematurely implement distributed training.

---

# 64. Observability

Every job must emit structured logs with:

```text
job_id
stage
block
duration_ms
count_in
count_out
model_version
error_code
```

Never log CAD file contents or sensitive geometry unless explicitly required for debugging.

---

# 65. Configuration

Create typed configuration.

At minimum:

```text
MAX_UPLOAD_BYTES
DEFAULT_UNIT
SNAP_TOLERANCE_M
INTERSECTION_TOLERANCE_M
GAP_MAX_DISTANCE_M
GAP_THRESHOLD
FACE_MIN_AREA_M2
ENTITY_THRESHOLD
FACE_THRESHOLD
OBJECT_ASSIGNMENT_TOLERANCE_M
JOB_TIMEOUT_SECONDS
RETRAIN_EVENT_THRESHOLD
```

Every configuration used by a job contributes to `configuration_hash`.

Never hard-code runtime thresholds in model code.

---

# 66. Security requirements

- Signed URLs for CAD files.
- No public object listing.
- Validate upload size.
- Validate file extension/content.
- Do not execute CAD-embedded scripts/macros.
- Sanitize file names.
- Store source files under generated object keys, not user-controlled paths.
- Admin training endpoint requires admin authorization once authentication is added.
- Consent flag controls whether training data may be used.

The source design explicitly calls out CAD IP, signed URLs, retention and training-improvement consent. fileciteturn0file0L321-L329

---

# 67. API upload behavior

`POST /jobs`:

```text
receive file
→ validate
→ persist source
→ create job
→ enqueue
→ return job_id
```

Never execute full CAD analysis inside the HTTP request.

`POST /jobs/from-url` follows:

```text
validate allowed R2 URL
→ create job
→ queue
```

The source design explicitly specifies asynchronous analysis with polling. fileciteturn0file0L94-L104

---

# 68. Worker behavior

Worker algorithm:

```python
load job
assert status == queued
transition to running
resolve model versions
load source
execute pipeline
validate LayoutDto
write snapshot
persist completion
transition to succeeded
```

On unrecoverable exception:

```python
persist structured failure
transition to failed
```

Do not swallow exceptions.

---

# 69. Pipeline context

Use a typed immutable-ish `PipelineContext` containing:

```text
job_id
engine_sha
configuration
model_versions
source_hash
timings
warnings
logger
```

Do not put mutable CAD state into a globally shared context.

---

# 70. No hidden side effects

These functions should be pure wherever possible:

```text
feature extraction
geometry normalization
polygonization
face classification
layout assembly
view reduction
metric calculation
```

I/O should be isolated to:

```text
store
object_store
queue
API
worker orchestration
trainer orchestration
```

---

# 71. What the coding agent must NOT do

Do NOT:

- revive the old `Scene`;
- keep process-memory jobs;
- make dividers the only training signal;
- drop faces permanently;
- use raw dictionaries in API responses;
- use Flask;
- add `extra="allow"` to public contracts;
- let GNNs create arbitrary room polygons;
- let a model change another job's model version;
- split datasets by face;
- train on implicit acceptance as strong labels;
- delete old model versions;
- mix coordinate units;
- silently repair invalid CAD without recording it;
- silently discard objects/texts;
- invent long gap links;
- couple opening/object logic directly into polygonization;
- introduce Kafka/message brokers in v1;
- split into microservices prematurely;
- build a frontend in this repository unless explicitly requested;
- change API v1 schemas casually.

---

# 72. Implementation order

## Phase P0 — Foundations

Implement:

```text
repository
configuration
contracts
FastAPI factory
OpenAPI
Postgres
Redis
object store abstraction
job table
health endpoint
```

### P0 definition of done

- `/docs` works.
- `/redoc` works.
- `/openapi.json` works.
- all DTOs validate.
- database migrations run from empty DB.
- queue connection works.
- object storage adapter works.
- OpenAPI snapshot test passes.
- health reports PostgreSQL/Redis/store status.

---

# 73. Phase P1 — Deterministic baseline

Implement:

```text
ingest
primitive extraction
unit normalization
primitive IDs
heuristic entity classifier
graphkit
portal detection
gap heuristic
polygonization
face features
face heuristic
room assembly
openings
objects
texts
warnings
snapshot
layout persistence
```

### P1 definition of done

A real DXF can:

```text
upload
→ queue
→ process
→ produce durable snapshot
→ return valid LayoutDto
```

No ML model is required for P1.

---

# 74. Phase P2 — Feedback lake

Implement:

```text
feedback DTOs
event persistence
event sequence
undo
layout revision
view reducer
divider endpoints
face listing
accept
compiler
parquet datasets
```

### P2 definition of done

A user action produces:

```text
HTTP request
→ validated event
→ durable event
→ new layout revision
→ deterministic derived layout
→ training row after compile
```

---

# 75. Phase P3 — GNN-C

Implement:

```text
face dataset
feature schema
PyG model
training loop
held-out evaluation
model registry
runtime
heuristic fallback
```

Gate production promotion on held-out quality.

The source plan explicitly says not to train P3 prematurely. fileciteturn0file0L368-L371

---

# 76. Phase P4 — GNN-A + GNN-B

Implement:

```text
entity dataset
entity GNN
gap dataset
gap GNN
runtime selection
hard-negative mining
model comparison
```

The objective is to improve:

```text
junk-wall rejection
missing-wall closure
```

while retaining deterministic polygonization.

---

# 77. Phase P5 — Structural modules

Implement/refine:

```text
door detector
window detector
object detector
text association
measurement association
```

These remain independent modules.

---

# 78. Phase P6 — Trainer operations

Implement:

```text
scheduled retraining
manual training endpoint
candidate registry
shadow mode
promotion
rollback
metrics history
```

The source plan defines P6 around the trainer app, registry, shadow promotion and rollback. fileciteturn0file0L365-L369

---

# 79. Agent execution protocol

The coding agent must work phase-by-phase.

For each phase:

1. inspect existing repository;
2. identify conflicts with this plan;
3. implement only the current phase;
4. add tests;
5. run formatting/lint/type checks;
6. run unit tests;
7. run integration tests relevant to the phase;
8. verify OpenAPI if API changed;
9. produce a short implementation report;
10. only then move to the next phase.

Do not implement future phases early unless a dependency requires a minimal placeholder.

---

# 80. Agent decision protocol

When encountering an ambiguity:

### First priority

Existing contract in this document.

### Second priority

Existing source architecture explicitly retained by the source design.

### Third priority

Minimal deterministic implementation.

### Never

Invent a different architecture just because it is fashionable or easier.

---

# 81. Required implementation reports

At the end of every phase the coding agent must report:

```text
implemented:
changed files:
database migrations:
API changes:
tests:
commands executed:
known limitations:
next phase:
```

For every failed test, report the root cause.

Do not claim a phase is complete if tests are failing.

---

# 82. Definition of "done"

The project is not considered implemented because the code imports successfully.

A feature is done only when:

```text
contract exists
implementation exists
persistence exists if required
tests exist
error behavior exists
observability exists where applicable
documentation/OpenAPI exists where applicable
regression test exists for important behavior
```

---

# 83. Final system invariant

At runtime the architecture must always satisfy:

```text
CAD
 ↓
Document
 ↓
PrimitiveSet
 ↓
PlanGraph
 ↓
FaceSet
 ↓
Physical Layout
 ↓
Immutable Snapshot
 ↓
Derived User View
```

And the learning loop must always satisfy:

```text
Snapshot + Explicit Events
        ↓
    Training Lake
        ↓
      Trainer
        ↓
       Eval
        ↓
     Registry
        ↓
Pinned Model Versions
        ↓
      Runtime
```

No user edit directly mutates the detector's source graph.

No model directly owns geometry.

No API request bypasses the contract layer.

No successful analysis exists without a durable snapshot.

---

# 84. Final coding-agent instruction

Treat this document as the implementation contract.

Do not rewrite the architecture while implementing it.

Prioritize correctness and deterministic behavior over premature optimization.

When a model is unavailable, the system must still work using the heuristic implementation behind the same interface.

When feedback is submitted, persist it as an event before producing the derived layout.

When a job succeeds, its snapshot must be durable and reproducible.

When a model is promoted, existing jobs must remain reproducible.

When a face is dropped from the user-visible room list, it must remain available to training.

When a user splits or merges rooms, modify only the derived view.

When a GNN is uncertain, fall back safely and record the uncertainty.

The result must be a working hybrid CAD engine first, and a continuously improving ML system second.
