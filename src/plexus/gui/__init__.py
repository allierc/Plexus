"""Plexus GUI -- an interactive node editor for `spec.yaml`.

A Plexus spec IS a node graph: operators are boxes (typed ports = the state they
read/write + the sets/fields they act on), sets and fields are the other node
kinds, and the `schedule` is the execution rail. This package renders that graph
in the browser (flat, dark, vertex-techno), driven by the operator registry, and
round-trips edits back to a validated `spec.yaml`.

    python -m plexus.gui [spec.yaml | dir]   # launch; opens the browser

Backend is stdlib-only (`http.server`); the frontend is vanilla JS/SVG in
`static/`.
"""

from plexus.gui.server import serve
from plexus.gui.catalog import build_catalog

__all__ = ["serve", "build_catalog", "launch"]


def launch(spec=None, host="127.0.0.1", port=8765, open_browser=True):
    """Start the editor server (blocking). Optionally deep-link to a spec."""
    from plexus.gui.__main__ import main
    argv = []
    if spec:
        argv.append(spec)
    argv += ["--host", host, "--port", str(port)]
    if not open_browser:
        argv.append("--no-browser")
    main(argv)
