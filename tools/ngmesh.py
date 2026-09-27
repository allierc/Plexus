"""Readers for the neuroglancer precomputed meshes of Janelia OpenOrganelle (legacy manifests, and
neuroglancer_multilod_draco). Written 2026-09-26 for the 3D scene data of experiments 6-15."""
import json, os, struct, numpy as np
def legacy(path):
    b = open(path, "rb").read()
    if b[:1] == b"{":
        vs, fs, off = [], [], 0
        for fr in json.loads(b)["fragments"]:
            p = os.path.join(os.path.dirname(path), fr)
            if os.path.exists(p):
                v, f = legacy(p); vs.append(v); fs.append(f + off); off += len(v)
        return (np.concatenate(vs), np.concatenate(fs)) if vs else (np.zeros((0, 3)), np.zeros((0, 3), int))
    n = struct.unpack("<I", b[:4])[0]
    return np.frombuffer(b[4:4 + 12 * n], "<f4").reshape(-1, 3), np.frombuffer(b[4 + 12 * n:], "<u4").reshape(-1, 3)
def multilod_draco(dirpath, seg, lod=None):
    """neuroglancer_multilod_draco: decode one segment at one level of detail (default the finest, 0)."""
    import DracoPy
    info = json.load(open(os.path.join(dirpath, "info"))); bits = info["vertex_quantization_bits"]
    b = open(os.path.join(dirpath, f"{seg}.index"), "rb").read(); o = 0
    rd = lambda fmt, n: (struct.unpack_from(f"<{n}{fmt}", b, o), o + 4 * n)
    chunk, o = rd("f", 3); origin, o = rd("f", 3); (nl,), o = rd("I", 1)
    scales, o = rd("f", nl); voff = np.array(struct.unpack_from(f"<{3*nl}f", b, o)).reshape(nl, 3); o += 12 * nl
    nfr, o = rd("I", nl)
    data = open(os.path.join(dirpath, str(seg)), "rb").read(); pos = 0; lod = 0 if lod is None else lod
    vs, fs, off = [], [], 0
    for l in range(nl):
        k = nfr[l]; fp = np.array(struct.unpack_from(f"<{3*k}I", b, o)).reshape(3, k).T; o += 12 * k
        sz = struct.unpack_from(f"<{k}I", b, o); o += 4 * k
        for j in range(k):
            blob = data[pos:pos + sz[j]]; pos += sz[j]
            if l != lod or sz[j] == 0: continue
            m = DracoPy.decode(blob); q = np.asarray(m.points, float); f = np.asarray(m.faces, int).reshape(-1, 3)
            v = np.array(origin) + voff[l] + np.array(chunk) * (2 ** l) * (fp[j] + q / (2 ** bits - 1))
            vs.append(v); fs.append(f + off); off += len(v)
    return (np.concatenate(vs), np.concatenate(fs)) if vs else (np.zeros((0, 3)), np.zeros((0, 3), int))
