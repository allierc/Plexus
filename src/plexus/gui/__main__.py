"""`python -m plexus.gui [--tab bio|material|neurons|metabolism] [--port N] [--no-browser]`.

Starts the server and (unless --no-browser) opens the one page on the tab asked for.
`--editor` opens the spec node editor instead (`/editor`).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from urllib.parse import quote

from plexus.gui.server import serve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plexus.gui", description="the Plexus page")
    ap.add_argument("spec", nargs="?", help="spec.yaml to open in the editor (with --editor)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799, help="default 8799, so the URL can be bookmarked")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--tab", default="material", choices=("bio", "material", "neurons", "metabolism"),
                    help="the tab the page opens on")
    ap.add_argument("--editor", action="store_true", help="open the spec node editor instead of the page")
    args = ap.parse_args(argv)

    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/" + ("editor" if args.editor else f"?tab={args.tab}")
    if args.spec and args.editor:
        sp = os.path.abspath(os.path.expanduser(args.spec))
        if os.path.isfile(sp):
            url += f"?spec={quote(sp)}"

    print(f"  {'Plexus spec editor' if args.editor else 'Plexus'}  ->  {url}")
    print("  (Ctrl-C to stop)")

    if not args.no_browser:
        threading.Timer(0.6, lambda: _try_open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.shutdown()


def _try_open(url):
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 -- headless / no browser; the URL was printed
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
