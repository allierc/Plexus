# run_03 — reading the Blender model, and looking at the muscles

`../../260802_s2_EYE_MUSCLES_MODEL.blend` (Blender 4.0, 28 MB) opened in Python and cut
into named parts: **6 muscles × 2 sides, 3 eye tissues × 2 sides**, plus the cartilage
and the CNS.

```bash
python read_blend.py --figure --ply       # cut it; re-execs itself into the bpy venv
python view_muscles_vtk.py --turntable 72 # 3-D view of the twelve muscles
```

## What the blend actually contains

Only the animal's LEFT half is modelled: every eye part and every muscle carries a
MIRROR modifier across x = 0, so the right eye and its six muscles exist only in the
evaluated mesh. `read_blend.py` evaluates the depsgraph, splits each mesh into loose
parts and labels a part by the sign of its centroid's x. Subsurf (viewport level 1) is
switched OFF by default, so a muscle comes back as the artist's 1120-vertex cage.

The muscles are called `Cylinder.001..006` in the file. They are renamed **from
measured geometry**, never from the object names or the material colours:

| Blender | key | insertion, in the eye frame (caudal, dorsal, lateral) | contact with the globe (insertion end / origin end) |
|---|---|---|---|
| Cylinder.006 | LR | ( 0.976,  0.179,  0.126) — caudal sclera | 0.041 / 0.273 |
| Cylinder.003 | SR | ( 0.314,  0.944, −0.101) — dorsal | 0.104 / 0.300 |
| Cylinder.004 | MR | (−0.952,  0.159, −0.263) — anteromedial | 0.079 / 0.289 |
| Cylinder.001 | IR | (−0.172, −0.985,  0.004) — ventral | 0.065 / 0.278 |
| Cylinder.005 | SO | (−0.352,  0.931, −0.099) — dorsal, origin at y = 1.20 | 0.068 / 0.572 |
| Cylinder.002 | IO | (−0.835, −0.505,  0.218) — ventral, origin at y = 1.30 | 0.048 / 0.585 |

Which end of a strap is the INSERTION is decided by distance to the retina shell — an
insertion end lies 0.04–0.10 world units off it, an origin end 0.27–0.59. The obvious
alternative, "the end nearest a sphere of the globe's mean radius", is ambiguous and
gets LR and IR backwards, because the caudal plate that SR/IR/MR/LR arise from happens
to sit about one globe radius (0.57) from the globe's centre. The two most rostral
origins (y = 1.20 and 1.30, against 2.6–2.9 for the recti) are the obliques — they arise
together from the anterior ethmoid plate, which is what `fish_anatomy` describes.

The mapping is written to `blend_parts/muscle_names.json` and that file WINS on the next
run: edit it to override any call without touching the code.

## Frames

Head axes, as in the blend: +x the animal's right, +y caudal, +z dorsal. Per eye we also
build `fish_anatomy`'s frame — (caudal, dorsal, LATERAL = the optic axis, measured centre
→ cornea, tilted ~15° rostrally). The eyes are enantiomorphs, so that triad is
left-handed on the left (`frame_det = −1`) and right-handed on the right (`+1`): the two
sides give the same insertion coordinates and OPPOSITE rotation-axis signs, which is the
sign flip a downstream torsion convention has to carry.

Globe: centre (∓0.862, 2.163, −0.133), semi-axes (0.688, 0.575, 0.468), mean radius 0.571.

## Outputs

| file | what |
|---|---|
| `blend_parts/parts.npz` | `<part>__v` [n,3] world verts, `<part>__f` [m,3] triangles, `<muscle>__centreline__v` [24,3] |
| `blend_parts/parts.json` | per part: group, side, counts, centroid, bbox, volume; per muscle: origin, insertion, length, line of action, rotation axis; per side: globe fit + frame |
| `blend_parts/muscle_names.json` | `Cylinder.00N → key`, re-read on the next run |
| `blend_parts/meshes/*.ply` | one PLY per part (`--ply`) |
| `blend_parts/blend_parts.png` | 3-view check figure of the whole cut (`--figure`) |
| `muscles_3d.png`, `muscles_3d_turntable.mp4` | the muscles in 3-D, from `view_muscles_vtk.py` |

Part names: `L_LR … R_IO`, `L_cornea`/`L_lens`/`L_retina` (+ `R_`), `cns`,
`bone_basal_plate_029_L`, …

## The two interpreters

Reading a .blend needs Blender's own loader, so `read_blend.py` runs under
`bpy==5.2.0` in `/workspace/.conda_envs/bpy-env` (cp313 wheel; Blender 5.2 reads the 4.0
file). It re-execs itself there, and points `LD_LIBRARY_PATH` at a conda env that has
libXfixes — bpy links X libraries this container does not install system-wide. The
viewer needs none of that: it reads the npz with pyvista in the ordinary project env.
