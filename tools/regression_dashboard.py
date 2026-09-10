"""The regression dashboard: every run in the archive, in a browser, with run-to-run comparison.

    python tools/regression_dashboard.py                    serve log/regression/archive.jsonl on :8765 and open it
    python tools/regression_dashboard.py --port 8800 --no-browser
    python tools/regression_dashboard.py --archive other.jsonl

Standard library only. The page reads the archive on every refresh, so a run appended while the
server is up shows on reload. Three views: the matrix (working points x runs, coloured by status,
click a cell for its violations), the comparison (pick run A and run B: every checkpoint metric of
every working point side by side with the relative difference), and frame time per working point
across runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARCHIVE = os.path.join(ROOT, "log", "regression", "archive.jsonl")

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Plexus regression</title>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:18px;color:#222}
 h1{font-size:20px;margin:0 0 6px} h2{font-size:16px;margin:22px 0 8px}
 table{border-collapse:collapse;font-size:13px} th,td{border:1px solid #ddd;padding:3px 7px;text-align:left;vertical-align:top}
 th{background:#f4f4f4;position:sticky;top:0}
 .PASS{background:#d8f2d8}.DIFFER{background:#ffd9b3}.ERROR{background:#f5b7b1}.SKIP{background:#eee;color:#888}
 .cell{cursor:pointer;text-align:center;font-weight:600}
 .num{text-align:right;font-variant-numeric:tabular-nums}
 .bad{color:#b00020;font-weight:600} .ok{color:#1b7f3b}
 select{font-size:13px} #detail{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:8px;font-size:12px;min-height:40px}
 .small{color:#666;font-size:12px}
 svg text{font-size:10px}
</style></head><body>
<h1>Plexus regression</h1>
<div class="small" id="meta"></div>
<h2>Runs</h2><div id="runs"></div>
<h2>Working points &times; runs <span class="small">(click a cell)</span></h2><div id="matrix"></div>
<div id="detail">click a cell</div>
<h2>Compare two runs</h2>
<div>A: <select id="selA"></select> &nbsp; B: <select id="selB"></select> &nbsp;
 <label><input type="checkbox" id="onlyDiff"> only metrics outside the band</label></div>
<div id="compare"></div>
<h2>Frame time (ms/frame)</h2><div id="timing"></div>
<script>
const BANDS = {cells:.03,half_edges:.03,deaths_or_net_loss:.03,ndiv_sum:.05,r_med:.01,r_p10:.02,r_p90:.02,area_mean:.02,roughness:.25,z_sd:.25,mpm_spread:.05,mpm_bbox:.05,mpm_inside:.05,chem_max:.1,chem_min:.1};
let RUNS = [];
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function fmt(v){ if(v===null||v===undefined) return ''; if(typeof v==='number') return Number.isInteger(v)?v:v.toPrecision(5); return esc(v); }
function label(r){ return r.run_id+'  '+r.commit+(r.dirty?'*':'')+'  '+(r.gpu||'')+(r.quick?'  quick':''); }
async function load(){
  const res = await fetch('/data'); RUNS = await res.json();
  document.getElementById('meta').textContent = RUNS.length+' run(s) in the archive; newest first. * = uncommitted changes in src/config/tools/tests at run time.';
  const names = [...new Set(RUNS.flatMap(r=>Object.keys(r.working_points||{})))].sort();
  // runs table
  let h='<table><tr><th>run</th><th>commit</th><th>branch</th><th>host</th><th>GPU</th><th>pass</th><th>differ</th><th>error</th><th>skip</th><th>seed table</th><th>invariants</th><th>scaling</th><th>min</th><th>note</th></tr>';
  for(const r of RUNS){ const s=r.summary||{}; const st=(r.seed_table||{}).status||'';
    const inv=r.invariants?`${r.invariants.passed}/${r.invariants.passed+r.invariants.failed}`:''; const sc=r.scaling?`${r.scaling.passed}/${r.scaling.passed+r.scaling.failed}`:'';
    h+=`<tr><td>${r.run_id}</td><td>${r.commit}${r.dirty?'*':''}</td><td>${esc(r.branch||'')}</td><td>${esc(r.host||'')}</td><td>${esc(r.gpu||'')}</td><td class="num ok">${s.pass??''}</td><td class="num ${s.differ?'bad':''}">${s.differ??''}</td><td class="num ${s.error?'bad':''}">${s.error??''}</td><td class="num">${s.skip??''}</td><td class="${st}">${st}</td><td class="${r.invariants&&r.invariants.failed?'bad':''}">${inv}</td><td class="${r.scaling&&r.scaling.failed?'bad':''}">${sc}</td><td class="num">${r.duration_s?(r.duration_s/60).toFixed(1):''}</td><td>${esc(r.note||'')}</td></tr>`; }
  document.getElementById('runs').innerHTML=h+'</table>';
  // matrix
  h='<table><tr><th>working point</th>'+RUNS.map(r=>`<th title="${esc(label(r))}">${r.run_id.slice(5,16)}<br><span class="small">${r.commit}</span></th>`).join('')+'</tr>';
  names.forEach(n=>{ h+=`<tr><td>${n}</td>`; RUNS.forEach((r,i)=>{ const w=(r.working_points||{})[n]; if(!w){h+='<td></td>';return;}
      const ms=w.ms_per_frame?`<br><span class="small">${w.ms_per_frame.toFixed(0)} ms</span>`:''; h+=`<td class="cell ${w.status}" onclick="detail(${i},'${n}')">${w.status}${ms}</td>`; }); h+='</tr>'; });
  document.getElementById('matrix').innerHTML=h+'</table>';
  // selects
  const opts=RUNS.map((r,i)=>`<option value="${i}">${esc(label(r))}</option>`).join('');
  const A=document.getElementById('selA'), B=document.getElementById('selB'); A.innerHTML=opts; B.innerHTML=opts; if(RUNS.length>1) B.value=1;
  A.onchange=B.onchange=document.getElementById('onlyDiff').onchange=compare; compare(); timing(names);
}
function detail(i,n){ const r=RUNS[i], w=r.working_points[n]; let t=`${n}  --  ${label(r)}\nstatus ${w.status}   cut ${w.cut??''} frames   ${w.seconds??''} s   ${w.timing_note||''}\nreference: archived ${w.ref_archived_on||''}, fingerprint commit ${w.ref_commit||''}\n`;
  if(w.violations&&w.violations.length) t+='\nviolations:\n  '+w.violations.join('\n  ');
  if(w.checkpoints){ t+='\n\ncheckpoints (run | reference):'; for(const f of Object.keys(w.checkpoints)){ const g=w.checkpoints[f], rf=(w.ref_checkpoints||{})[f]||{}; t+=`\n  frame ${f}:`; for(const k of Object.keys(g)){ if(Array.isArray(g[k])||typeof g[k]==='string') continue; t+=`\n     ${k.padEnd(20)} ${fmt(g[k]).toString().padStart(12)} | ${fmt(rf[k]).toString().padStart(12)}`; } } }
  document.getElementById('detail').textContent=t; }
function compare(){ const a=RUNS[+document.getElementById('selA').value], b=RUNS[+document.getElementById('selB').value]; if(!a||!b) return;
  const only=document.getElementById('onlyDiff').checked;
  let h='<table><tr><th>working point</th><th>frame</th><th>metric</th><th class="num">A</th><th class="num">B</th><th class="num">B vs A</th><th>band</th></tr>';
  const names=[...new Set([...Object.keys(a.working_points||{}),...Object.keys(b.working_points||{})])].sort();
  for(const n of names){ const wa=(a.working_points||{})[n], wb=(b.working_points||{})[n]; if(!wa||!wb||!wa.checkpoints||!wb.checkpoints){ h+=`<tr><td>${n}</td><td colspan=6 class="small">${wa?wa.status:'absent'} / ${wb?wb.status:'absent'}</td></tr>`; continue; }
    if(wa.ms_per_frame&&wb.ms_per_frame){ const d=(wb.ms_per_frame-wa.ms_per_frame)/wa.ms_per_frame; if(!only||Math.abs(d)>0.5) h+=`<tr><td>${n}</td><td>-</td><td>ms/frame (${esc(a.gpu)} vs ${esc(b.gpu)})</td><td class="num">${wa.ms_per_frame.toFixed(1)}</td><td class="num">${wb.ms_per_frame.toFixed(1)}</td><td class="num ${Math.abs(d)>0.5?'bad':''}">${(d*100).toFixed(1)}%</td><td>50% (same GPU)</td></tr>`; }
    for(const f of Object.keys(wa.checkpoints)){ const ga=wa.checkpoints[f], gb=wb.checkpoints[f]||{}; for(const k of Object.keys(ga)){ if(!(k in BANDS)) continue; const va=ga[k], vb=gb[k]; if(vb===undefined) continue;
        const scale=(k==='deaths_or_net_loss'||k==='ndiv_sum')?Math.max((wa.checkpoints['0']||{}).cells||Math.abs(va),1):Math.max(Math.abs(va),1e-9); const d=(vb-va)/scale; const out=Math.abs(d)>BANDS[k];
        if(only&&!out) continue; h+=`<tr><td>${n}</td><td class="num">${f}</td><td>${k}</td><td class="num">${fmt(va)}</td><td class="num">${fmt(vb)}</td><td class="num ${out?'bad':''}">${(d*100).toFixed(2)}%</td><td class="small">${(BANDS[k]*100).toFixed(0)}%</td></tr>`; } } }
  document.getElementById('compare').innerHTML=h+'</table>'; }
function timing(names){ const W=900,H=260,L=60,Bm=40; let s=`<svg width="${W}" height="${H}" style="background:#fff;border:1px solid #ddd">`;
  const runs=[...RUNS].reverse(); const series=names.map(n=>({n, pts:runs.map((r,i)=>{const w=(r.working_points||{})[n]; return w&&w.ms_per_frame?[i,w.ms_per_frame,r.gpu]:null;})})).filter(sr=>sr.pts.some(p=>p));
  const ymax=Math.max(1,...series.flatMap(sr=>sr.pts.filter(p=>p).map(p=>p[1]))); const x=i=>L+(W-L-20)*(runs.length>1?i/(runs.length-1):0.5), y=v=>H-Bm-(H-Bm-15)*v/ymax;
  for(let t=0;t<=4;t++){const v=ymax*t/4; s+=`<line x1="${L}" x2="${W-20}" y1="${y(v)}" y2="${y(v)}" stroke="#eee"/><text x="4" y="${y(v)+3}">${v.toFixed(0)}</text>`;}
  runs.forEach((r,i)=>{ s+=`<text x="${x(i)}" y="${H-Bm+14}" text-anchor="middle">${r.run_id.slice(5,10)}</text><text x="${x(i)}" y="${H-Bm+26}" text-anchor="middle" fill="#888">${r.commit}</text>`; });
  series.forEach((sr,k)=>{ const col=`hsl(${(k*47)%360},60%,45%)`; const pts=sr.pts.map((p,i)=>p?`${x(p[0])},${y(p[1])}`:null).filter(Boolean); if(pts.length>1) s+=`<polyline fill="none" stroke="${col}" stroke-width="1.5" points="${pts.join(' ')}"/>`;
    sr.pts.forEach(p=>{ if(p) s+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${col}"><title>${sr.n}: ${p[1].toFixed(1)} ms on ${esc(p[2]||'')}</title></circle>`; }); const last=[...sr.pts].reverse().find(p=>p); if(last) s+=`<text x="${x(last[0])+5}" y="${y(last[1])+3}" fill="${col}">${sr.n}</text>`; });
  document.getElementById('timing').innerHTML=s+'</svg><div class="small">points on different GPUs are not comparable; hover a point for the card.</div>'; }
load();
</script></body></html>
"""


def read_archive(path: str) -> list[dict]:
    runs = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    runs.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return runs


def make_handler(archive: str):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data"):
                body = json.dumps(read_archive(archive)).encode()
                ctype = "application/json"
            else:
                body = PAGE.encode(); ctype = "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):                        # quiet
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", args.port), make_handler(args.archive))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[dashboard] {len(read_archive(args.archive))} run(s) from {args.archive}\n[dashboard] {url}  (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
