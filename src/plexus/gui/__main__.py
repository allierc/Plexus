"""`python -m plexus.gui [spec.yaml | dir] [--port N] [--host H] [--no-browser]`.

Starts the local editor server and (unless --no-browser) opens the page, deep-
linking to the spec if one was given.
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
    ap = argparse.ArgumentParser(prog="plexus.gui", description="Plexus spec node editor")
    ap.add_argument("spec", nargs="?", help="spec.yaml to open (or a dir to browse)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None,
                    help="default 8765; the bio page is always on 8799 so its URL can be bookmarked")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--studio", action="store_true",
                    help="open the prompt-to-scene studio instead of the node editor")
    ap.add_argument("--bio", action="store_true",
                    help="open the bio-objects page: define a tissue and its proteins, seed, click")
    ap.add_argument("--material", action="store_true",
                    help="open the material page: MPM bodies in a box, seed, run (always on 8798)")
    args = ap.parse_args(argv)

    if args.port is None:
        args.port = 8799 if args.bio else 8798 if args.material else 8765   # /bio on 8799, /material on 8798: bookmarks
    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/" + ("bio" if args.bio else "material" if args.material else "studio" if args.studio else "")
    if args.spec and not args.studio:
        sp = os.path.abspath(os.path.expanduser(args.spec))
        if os.path.isfile(sp):
            url += f"?spec={quote(sp)}"

    print(f"  {'Plexus bio objects' if args.bio else 'Plexus material' if args.material else 'Plexus Studio' if args.studio else 'Plexus spec editor'}  ->  {url}")
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
