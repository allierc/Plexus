"""vlm_server.py -- load Gemma-4-12B ONCE and serve it to every caller on this machine.

    python VLLM/vlm_server.py [--port 8766] [--device cuda:1]

WHY. A simulation-branch search (experiments/INSTRUCTION.md) asks the model two things per
iteration: the generate pipeline's caption pass (`describe_video.py`) and the judge's score
(`tools/judge.py`). Each loaded the 12B model from scratch, ~70 s apiece, around an engine run of
11 s. This process pays that load once; both callers find it through `vlm_client.server_url()` and
fall back to loading their own copy when it is not running, so nothing depends on it.

ONE REQUEST AT A TIME. `HTTPServer`, not the threading variant: generate calls on one model share
one GPU, and serialising them here is what a queue of callers would get anyway, without the risk
of two `generate`s interleaving on the same weights.

Routes (JSON in, JSON out):
    GET  /health                                         -> {model, device, served}
    POST /caption {video, frames=8, layout=None}         -> {text}   (describe_one: DESCRIPTION/OBJECTS)
    POST /ask     {images: [png paths], text, max_new_tokens=400} -> {text}
"""
import argparse
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import describe_video as dv                                     # noqa: E402


class Handler(BaseHTTPRequestHandler):
    proc = model = None
    device = ""
    served = 0

    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                                  # the per-request line below instead
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._reply(200, {"model": dv.GEMMA, "device": self.device, "served": Handler.served})
        else:
            self._reply(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        t0 = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
            if self.path == "/caption":
                text = dv.describe_one(self.proc, self.model, req["video"],
                                       int(req.get("frames", 8)), req.get("layout"))
                what = os.path.basename(os.path.dirname(req["video"])) + "/" + os.path.basename(req["video"])
            elif self.path == "/ask":
                text = dv.ask(self.proc, self.model, list(req["images"]), req["text"],
                              int(req.get("max_new_tokens", 400)))
                what = f"{len(req['images'])} image(s)"
            else:
                return self._reply(404, {"error": f"no route {self.path}"})
            Handler.served += 1
            print(f"[vlm] {self.path} {what}  {time.time() - t0:.1f} s", flush=True)
            self._reply(200, {"text": text})
        except Exception as e:                                  # noqa: BLE001
            traceback.print_exc()
            self._reply(500, {"error": f"{type(e).__name__}: {str(e)[:300]}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default=None, help="cuda:N; default = the card with the most free memory")
    a = ap.parse_args()
    t0 = time.time()
    Handler.proc, Handler.model, Handler.device = dv._load(a.device or dv.DEV)
    print(f"[vlm] {dv.GEMMA} loaded on {Handler.device} in {time.time() - t0:.0f} s; "
          f"serving on 127.0.0.1:{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
