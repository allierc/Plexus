<!-- pixeltracker -- append below; the driver merges this into campaign/analysis.md -->

# PixelTracker (order 23) — excavator note

Read: PyCoreSpecs.py:5311 (the spec, one param `track_medium`), the SWIG stub
CompuCell.py:6002 (`class PixelTrackerPlugin(CellGChangeWatcher)` — the decisive
fact), the attached-data classes PixelTracker/PixelTrackerData (5837, 5854), the
iterator CellPixelList (iterators.py:700), and the consumers in PySteppables.py
(get_cell_pixel_list:2465, get_copy_of_cell_pixels:2584, move_cell/delete_cell).

What it does: it materialises the inverse of the cell_field. The CPM lattice stores
site -> cell id; PixelTracker keeps a live per-cell set of the sites that id owns,
updated incrementally on every accepted pixel copy via field3DChange (erase pt from
old owner, insert into new owner). Delta H = 0 — it is NOT an energy plugin.

Surprised me: this is the plugin that reifies the ATLAS's own framing ("a cell is a
set of lattice sites sharing an id"). In CC3D that set is NOT primitive — you must
load a watcher to build the id->{sites} map; the lattice alone only gives you
site->id. In Plexus the set is the primitive, so PixelTracker looks like an
implementation of a representation Plexus gets for free. Whether the algebra needs
an explicit "index / inverse-map over a field partition" contract is the open
question I flagged in surprises — I did NOT resolve it (that is the normalizer's call).

Could NOT establish: I could not read the compiled C++ field3DChange body itself —
only the SWIG signatures and the Python consumers — so the exact erase/insert
ordering and the medium-guard branch are reconstructed from the SWIG API
(enableMediumTracker, getMediumPixelSet, trackingMedium) plus behaviour, not from
reading the .cpp. High confidence given CellGChangeWatcher semantics, but stated as
reconstruction. No ablation/evidence run exists for this mechanism (not among the
six with metrics.json). track_medium default False is a performance guard, verified
from the spec's conditional <TrackMedium/> emission (PyCoreSpecs.py:5333).

Re-verification pass (correction): the earlier entry claimed get_cell_pixel_list
RAISES when PixelTracker is absent. It does not — it silently returns None
(PySteppables.py:2473-2476). The AttributeError is raised only by the higher-level
wrappers get_copy_of_cell_pixels (2603), delete_cell (2534), and move_cell (469).
Fixed the entry's dependency surprise to reflect the silent-None low-level path (a
worse gotcha than a clean raise: a forgotten None-check surfaces as a confusing
NoneType error downstream). Also confirmed the container is concretely `pixelSet`
on the extra-attrib accessor (iterators.py:703/724). Everything else in the entry
checked out against source; status stays inspected.

Anchor re-verification (this pass): re-read PyCoreSpecs.py:5311 (class def line
correct — code_path unchanged) and located the real SWIG stub at
cc3d/cpp/CompuCell.py (not a bare CompuCell.py): class PixelTrackerPlugin at :6002,
PixelTrackerPlugin_field3DChange at :6010/:6011, enableMediumTracker at :6031, all
confirmed. iterators.py:700 (CellPixelList) and :724 (pixelSet.size) confirmed
verbatim. Corrected the paper_section anchor to the full checkable path
cc3d/cpp/CompuCell.py:6002. Prior non-acceptance was purely a merge-hygiene issue
(atlas_record.yaml touched directly) — the science was unchanged; edited only the
working copy this time.

Line-attribution correction (this pass): re-reading the consumers showed the raise
sites were mis-attributed in the dependency surprise. The verified truth:
get_copy_of_cell_pixels (def 2584, raises 2603) and move_cell (def 2509, raises
2534) both raise AttributeError('Could not find PixelTracker Plugin');
delete_cell (def 2636) raises only INDIRECTLY through get_copy_of_cell_pixels; and
line 469 is merge_cells (def 458), which raises its own Exception "…did you load
PixelTracker plugin?" — NOT move_cell. Entry's third surprise updated to match.
