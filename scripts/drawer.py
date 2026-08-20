import json
import matplotlib
matplotlib.use('QtAgg')  # Or 'QtAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import urllib.request

job_id = "b3953bc1-bcb7-4d74-9eca-3a064f675bfb"
layout = json.load(urllib.request.urlopen(f"http://localhost:8000/api/v1/jobs/{job_id}/layout"))["data"]

fig, ax = plt.subplots()
for room in layout["rooms"]:
    pts = [(p["x"], p["y"]) for p in room["boundary"]]
    ax.add_patch(Polygon(pts, closed=True, alpha=0.3))
    c = room["center"]
    ax.text(c["x"], c["y"], room.get("name") or room["kind"], ha="center")

for w in layout["walls"]:
    ax.plot([w["start"]["x"], w["end"]["x"]], [w["start"]["y"], w["end"]["y"]], "k-")
for d in layout["doors"]:
    ax.plot([d["start"]["x"], d["end"]["x"]], [d["start"]["y"], d["end"]["y"]], "r-", lw=3)
for w in layout["windows"]:
    ax.plot([w["start"]["x"], w["end"]["x"]], [w["start"]["y"], w["end"]["y"]], "c-", lw=2)

ax.set_aspect("equal")
ax.set_xlabel("m")
ax.set_ylabel("m")

plt.show()