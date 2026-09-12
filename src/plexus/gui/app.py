"""The one page: a tab bar, the tab's form, and the shared panel under it.

`page(tab)` renders `/?tab=<name>`. The form is the tab's (`gui/tabs/<name>.py`: FORM_HTML,
FORM_JS); everything below it is here, once: OPEN, BUILD + SEED, the run (which is
`plexus.pipeline.generate` on the server's VTK thread -- the body of `Plexus_Main.py -o generate`,
with its per-frame hook feeding this page's renderer so orbit and zoom work while the movie is
written), PLAY (the run's kept frames at any camera), MOVIE (the mp4 it wrote), YAML, Claude, the
hierarchy and the selected object. Switching tabs re-initialises: the run is stopped, the view
dropped, the server's state cleared, and the new tab's default scene built and seeded on load.

The visual language is ngp-demo's: black ground, white ink, 1 px borders, small uppercase
letterspaced labels. No framework, no build step -- the page is a string and the JS is inline,
so the whole UI is greppable from this file and the tab files.
"""
from __future__ import annotations

from plexus.gui import tabs

CSS = r"""
 html,body{margin:0;height:100%;background:#0b0b0d;color:#ddd;font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
 #left{position:absolute;left:0;top:0;bottom:0;width:430px;overflow:auto;background:#141418;border-right:1px solid #2a2a30;padding:10px 12px;box-sizing:border-box}
 #right{position:absolute;left:430px;top:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
 #view{max-width:100%;max-height:100%;cursor:grab;user-select:none;-webkit-user-drag:none}
 #tabs{display:flex;gap:2px;margin:0 0 10px} #tabs a{flex:1;text-align:center;padding:6px 0;background:#1c1c22;color:#9ab;text-decoration:none;letter-spacing:.08em;text-transform:uppercase;font-size:11px;border:1px solid #2a2a30;cursor:pointer}
 #tabs a.on{background:#2b5f9e;color:#fff;border-color:#2b5f9e}
 h1{font-size:15px;margin:2px 0 8px;color:#fff} h2{font-size:12px;margin:14px 0 4px;color:#9ab;letter-spacing:.06em;text-transform:uppercase}
 label{display:inline-block;width:92px;color:#aab} input,select{background:#0e0e12;color:#eee;border:1px solid #333;border-radius:3px;padding:2px 5px;width:110px;margin:1px 0}
 input.short{width:56px} select{width:118px}
 button{background:#2b5f9e;color:#fff;border:0;border-radius:3px;padding:5px 10px;margin:3px 3px 3px 0;cursor:pointer} button.dim{background:#3a3a44}
 button:disabled{opacity:.5;cursor:default}
 table.sp{border-collapse:collapse;width:100%} table.sp td{padding:1px 2px} table.sp th{font-weight:normal;color:#889;font-size:11px} table.sp input,table.sp select{width:100%;box-sizing:border-box;font-size:11px}
 #status{color:#8c8;min-height:16px;margin:4px 0;white-space:pre-wrap;font-size:12px} #status.err{color:#f88}
 #tree div{padding:1px 0 1px 8px;cursor:pointer} #tree div:hover{color:#fff} #tree .n{color:#7fb3ff} #tree .rel{color:#9ac} #tree .cont{color:#c9a}
 #info{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:12px;white-space:pre-wrap;min-height:60px;max-height:260px;overflow:auto}
 textarea{width:100%;height:120px;background:#0e0e12;color:#ddd;border:1px solid #333;font:11px monospace;box-sizing:border-box}
 #yaml{display:none} .row{margin:2px 0}
 #hint{position:absolute;right:12px;top:8px;color:#778;font-size:12px}
 #vis label{width:auto;color:#ccd;margin-right:10px;cursor:pointer} #vis input[type=checkbox]{width:auto;margin:0 3px 0 0}
 button.claude{background:#000;border:1px solid #555;display:inline-flex;align-items:center;gap:6px} button.claude.on{background:#1f8f3f;border-color:#2fbf5f}
 button.claude svg{width:14px;height:14px;fill:#d97757} button.claude.on svg{fill:#fff}
 #claude{background:#0e0e12;border:1px solid #2a2a30;padding:6px;font-size:11px;white-space:pre-wrap;height:190px;overflow:auto;margin:4px 0}
"""

SHELL = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>
<div id="left">
 <div id="tabs">{tabbar}</div>
 <h2>Open a spec</h2>
 <div class="row"><button class="dim" onclick="pickOpen()">OPEN...</button> <span id="openlab" style="color:#778;font-size:11px">a spec.yaml or a run folder; copied into config/studio and seeded as is</span></div>
 <div id="picker" style="display:none;position:fixed;left:60px;top:40px;width:560px;max-height:80vh;background:#1a1a20;border:1px solid #556;border-radius:6px;padding:10px;z-index:10;box-shadow:0 0 30px #000">
  <div class="row"><b>Open a spec</b> <span style="float:right;cursor:pointer" onclick="$('picker').style.display='none'">&#10005;</span></div>
  <div class="row" id="pickroots"></div>
  <div class="row" id="pickpath" style="color:#9ab;font-family:monospace;font-size:11px;word-break:break-all"></div>
  <div id="picklist" style="max-height:55vh;overflow:auto;background:#0e0e12;border:1px solid #2a2a30;padding:4px;font-size:12px"></div>
 </div>
{form}
 <div class="row"><button onclick="build()">BUILD + SEED</button><button class="dim" onclick="toggleYaml()">YAML</button><button class="dim" onclick="reseed()">RE-SEED</button></div>
 <div id="status">building the default scene...</div>
 <h2>Run <span style="color:#778;font-weight:normal;text-transform:none">the same -o generate as Plexus_Main.py: movie and stills land in graphs_data/studio/&lt;name&gt;</span></h2>
 <div class="row"><label>device</label><select id="run_device" style="width:80px"><option>cuda:0</option><option>cuda:1</option><option>cpu</option></select> <span style="color:#778;font-size:11px">frames, movie frames and stills are the spec's (BUILD writes them)</span></div>
 <div class="row"><button onclick="runGo()" id="runbtn">RUN</button><button class="dim" onclick="runStop()">STOP</button> <span id="runstat" style="color:#8c8"></span></div>
 <div id="runcounts" style="color:#9ab;font-size:12px;min-height:14px"></div>
 <div class="row"><button class="dim" onclick="playGo()" id="playbtn">PLAY</button><button class="dim" onclick="playStop()">PAUSE</button><button class="dim" onclick="movieGo()" title="the movie.mp4 the run wrote, as a file">MOVIE</button><button class="dim" onclick="playStop();showMovie(null);FRAME=null;render(true)" title="back to the live view">LIVE</button> <input type="range" id="frame" min="0" max="0" value="0" style="width:150px" oninput="playing=null;showMovie(null);showFrame(+this.value)"> <span id="framelab" style="color:#9ab"></span></div>
 <h2>Claude</h2>
 <div class="row"><input id="task" style="width:100%" placeholder="{placeholder}" onkeydown="if(event.key==='Enter')claudeGo()"></div>
 <div class="row"><button onclick="claudeGo()" id="cbtn" class="claude"><svg viewBox="0 0 24 24"><path d="M12 1.5l1.6 6.4 5.6-3.6-3.6 5.6 6.4 1.6-6.4 1.6 3.6 5.6-5.6-3.6L12 22.5l-1.6-6.4-5.6 3.6 3.6-5.6L1.5 12l6.9-1.6-3.6-5.6 5.6 3.6z"/></svg>CLAUDE</button><button class="dim" onclick="claudeStop()">STOP</button><button class="dim" onclick="claudeNew()" title="forget the conversation so far">NEW SESSION</button> <span id="cstat" style="color:#8c8"></span></div>
 <pre id="claude"></pre>
 <div id="rstat" style="color:#9ab;min-height:14px"></div>
 <div id="yaml"><textarea id="yamltext"></textarea><div><button onclick="saveYaml()">SAVE YAML</button></div></div>
 <h2>Visibility</h2><div id="vis">(seed a scene first)</div>
 <h2>Hierarchy</h2><div id="tree">(none)</div>
 <h2>Selected object</h2><div id="info">click an object</div>
</div>
<div id="right"><img id="view" draggable="false"><video id="movie" style="display:none;max-width:100%;max-height:100%" controls muted></video><div id="hint">drag to orbit, wheel to zoom, click to select -- rendered by the movie renderer, also while a run is going</div></div>
<script>
const TAB={tab_json};
const DEFAULT_BODIES={default_bodies};
const $=id=>document.getElementById(id);
let SCENE=null, specName=null;
const CAM={{azim:30,elev:20,zoom:1}};
function status(t,err){{const s=$('status');s.textContent=t;s.className=err?'err':'';}}
{form_js}
function form(){{return tabForm();}}
function fillForm(f){{tabFill(f);}}
// SWITCHING TABS RE-INITIALISES: the server stops the run, drops the view and forgets the spec;
// the new tab's page then builds and seeds its default scene on load.
window.switchTab=async function(name){{if(name===TAB)return;try{{await post('/api/scene/reset',{{tab:name}});}}catch(e){{}}location.href='/?tab='+name;}};
// THE PICKER IS THE SERVER'S LISTING: a browser file dialog hands the page bytes, never a path, and
// the specs live where the server runs. Folders that hold a spec.yaml (run archives) open as one.
window.pickOpen=async function(path){{$('picker').style.display='block';const j=await (await fetch('/api/scene/ls?path='+encodeURIComponent(path||'{pick_dir}'))).json();if(j.error){{$('picklist').textContent=j.error;return;}}
 $('pickroots').innerHTML=Object.entries(j.roots).map(([k,v])=>`<button class="dim" onclick="pickOpen('${{v}}')">${{k}}</button>`).join('');$('pickpath').textContent=j.path;
 let h=`<div style="cursor:pointer;color:#9ac" onclick="pickOpen('${{j.parent}}')">.. (up)</div>`;
 for(const d of j.dirs)h+=`<div style="cursor:pointer;padding:1px 0"><span style="color:#7fb3ff" onclick="pickOpen('${{j.path}}/${{d.name}}')">&#128193; ${{d.name}}/</span>${{d.spec?` <button class="dim" style="padding:1px 6px;font-size:11px" onclick="openSpec('${{j.path}}/${{d.name}}')">open run</button>`:''}}</div>`;
 for(const f of j.files)h+=`<div style="cursor:pointer;padding:1px 0;color:#dde" onclick="openSpec('${{j.path}}/${{f}}')">&#128196; ${{f}}</div>`;
 $('picklist').innerHTML=h||'(empty)';}};
window.openSpec=async function(pth){{$('picker').style.display='none';status('opening '+pth+' ...');const j=await (await fetch('/api/scene/open?path='+encodeURIComponent(pth))).json();if(j.error){{status(j.error,true);return;}}$('openlab').textContent=pth;status('opened '+j.name+' -- seeding...');}};
async function post(url,body){{const r=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});return r.json();}}
window.build=async function(){{status('building the spec...');const j=await post('/api/tab/'+TAB+'/build',form());if(j.error){{status(j.error+(j.detail?'\n'+j.detail:''),true);return;}}specName=j.name;if(j.version!==undefined)seen.version=j.version;$('yamltext').value=j.raw;status(`spec saved: config/studio/${{j.name}}.yaml -- seeding and rendering...`);await reseed();}};
window.reseed=async function(){{playStop();showMovie(null);FRAME=null;if(!specName){{status('no spec yet',true);return;}}status('seeding and building the renderer...');const r=await fetch('/api/scene/seed?name='+encodeURIComponent(specName));const j=await r.json();if(j.error){{status(j.error,true);return;}}SCENE=j;visPanel(j);tree(j);status(`seeded in ${{j.seconds}}s: `+Object.entries(j.sets).map(([k,v])=>`${{k}} ${{v.n_live}}`).join(', '));render(true);}};
window.toggleYaml=function(){{const y=$('yaml');y.style.display=y.style.display==='none'?'block':'none';}};
window.saveYaml=async function(){{const j=await post('/api/scene/save',{{name:specName,raw:$('yamltext').value,tab:TAB}});if(j.error){{status(j.error+(j.detail?'\n'+j.detail:''),true);return;}}if(j.form)fillForm(j.form);status('saved; seeding...');await reseed();}};
// THE PICTURE IS THE MOVIE RENDERER'S. Every camera change asks the server for a fresh screenshot;
// at most one request is in flight and the newest camera wins, so dragging never queues up.
let inflight=false, dirty=false;
async function render(force){{if(inflight){{dirty=true;return;}}inflight=true;try{{const r=await fetch(`/api/scene/render?azim=${{CAM.azim}}&elev=${{CAM.elev}}&zoom=${{CAM.zoom}}${{FRAME===null?'':'&frame='+FRAME}}&t=${{Date.now()}}`);if(r.ok){{const b=await r.blob();const u=URL.createObjectURL(b);const im=$('view');const old=im.src;im.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}}else if(force){{status((await r.json()).error||'render failed',true);}}}}catch(e){{}}finally{{inflight=false;if(dirty){{dirty=false;render();}}}}}}
const im=$('view');let drag=null;
im.addEventListener('mousedown',e=>{{drag={{x:e.clientX,y:e.clientY,moved:false}};im.style.cursor='grabbing';}});
window.addEventListener('mousemove',e=>{{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.moved=true;CAM.azim-=dx*0.4;CAM.elev=Math.max(-89,Math.min(89,CAM.elev+dy*0.4));drag.x=e.clientX;drag.y=e.clientY;render();}});
window.addEventListener('mouseup',async e=>{{if(!drag)return;const moved=drag.moved;drag=null;im.style.cursor='grab';if(moved)return;
 const r=im.getBoundingClientRect();const fx=(e.clientX-r.left)/r.width,fy=(e.clientY-r.top)/r.height;if(fx<0||fx>1||fy<0||fy>1)return;
 const j=await (await fetch(`/api/scene/pick?x=${{fx.toFixed(4)}}&y=${{fy.toFixed(4)}}`)).json();if(j.pick){{showInfo(j.info);render();}}else{{$('info').textContent='nothing under the click';}}}});
im.addEventListener('wheel',e=>{{e.preventDefault();CAM.zoom=Math.max(0.05,Math.min(60,CAM.zoom*(e.deltaY>0?1/1.08:1.08)));render();}},{{passive:false}});
function showInfo(i){{if(!i){{$('info').textContent='(no object)';return;}}let t='';
 if(i.kind==='cell'){{t+=`cell #${{i.cell}}\n`;for(const [b,v] of Object.entries(i.blocks||{{}}))t+=`  ${{b}}: ${{JSON.stringify(v)}}\n`;for(const [s,per] of Object.entries(i.contains||{{}}))t+=`  contains ${{s}}: ${{Object.entries(per).map(([a,b])=>b+' '+a).join(', ')}}\n`;if(i.vertices)t+=`  vertices: ${{i.vertices.length}}`;}}
 else if(i.kind==='vertex'){{t+=`vertex #${{i.vertex}}\n  position: ${{i.position.join(', ')}}\n  shared by cells: ${{(i.shared_by_cells||[]).join(', ')}}`;}}
 else{{t+=`${{i.species||i.set}} #${{i.index}} (set ${{i.set}})\n  position: ${{(i.position||[]).map(v=>v.toFixed?v.toFixed(4):v).join(', ')}}\n`+(i.parent_cell!==undefined?`  parent: #${{i.parent_cell}}\n`:'');if(i.cell){{const c=i.cell;for(const [s,per] of Object.entries(c.contains||{{}}))t+=`  the parent contains ${{s}}: ${{Object.entries(per).map(([a,b])=>b+' '+a).join(', ')}}\n`;}}}}
 $('info').textContent=t;}}
function visPanel(j){{const sps=[];for(const s of Object.values(j.sets))for(const n of (s.type_names||[]))if(!sps.includes(n))sps.push(n);
 if(!sps.length){{$('vis').textContent='(no typed set)';return;}}
 $('vis').innerHTML='<div class="row">'+sps.map(n=>{{const c=((j.colors||{{}})[n]||[1,1,1]).map(x=>Math.round(x*255));return `<label><input type="checkbox" checked onchange="setVisible('${{n}}',this.checked)"><span style="color:rgb(${{c}})">&#9679;</span> ${{n}}</label>`;}}).join('')+'</div><div style="color:#778;font-size:11px">a type drawn as glyphs is one actor of the renderer; unticking hides it</div>';}}
window.setVisible=async function(n,on){{await post('/api/scene/visible',{{species:n,on}});render();}};
function tree(j){{const h=j.hierarchy;let out='';for(const n of h.sets){{const cnt=j.sets[n.name]?`${{j.sets[n.name].n_live}} live / ${{j.sets[n.name].n_buffer}}`:'';
 let rel='';if(n.parent)rel+=` <span class="cont">contained in ${{n.parent}}</span>`;if(n.maps)rel+=` <span class="rel">relation: ${{Object.entries(n.maps).map(([k,v])=>k+'->'+v).join(', ')}}</span>`;if(n.mesh)rel+=` <span class="rel">mesh: ${{n.mesh}}</span>`;
 out+=`<div onclick="window.setInfo('${{n.name}}')"><span class="n">${{n.name}}</span> ${{cnt}}${{rel}}${{n.types.length?' <span class="cont">types: '+n.types.join(', ')+'</span>':''}}</div>`;}}
 out+=`<div style="color:#778;margin-top:4px">schedule: ${{h.schedule.map(x=>typeof x==='string'?x:'{{substeps: '+(x.steps||[]).join(' > ')+'}}').join(' > ')}}</div>`;$('tree').innerHTML=out;}}
window.setInfo=function(name){{const s=SCENE.sets[name];const n=SCENE.hierarchy.sets.find(x=>x.name===name);
 $('info').textContent=`set ${{name}}\n  entity: ${{n.entity||'(by name)'}}\n  buffer ${{s.n_buffer}}, live ${{s.n_live}}\n  blocks: ${{s.blocks.join(', ')}}\n`+(n.parent?`  contained in: ${{n.parent}} (${{n.per_parent??'?'}} per parent)\n`:'')+(n.maps?`  relation maps: ${{JSON.stringify(n.maps)}}\n`:'')+(n.types.length?`  types: ${{n.types.join(', ')}}\n`:'')+`  operators on it: ${{SCENE.hierarchy.operators.filter(o=>o.at===name).map(o=>o.op).join(', ')||'-'}}`;}};
tabInit();
// FOLLOW THE SERVER'S SESSION: whoever drives the API (a person, or Claude) is seen here.
let seen={{version:-1,cam_version:-1}};
async function poll(){{try{{const st=await (await fetch('/api/scene/state')).json();
 if(st.name&&st.version!==seen.version){{seen.version=st.version;specName=st.name;$('name').value=st.name;const j=await (await fetch('/api/scene/spec?name='+encodeURIComponent(st.name)+'&tab='+TAB)).json();if(j.raw)$('yamltext').value=j.raw;if(j.form)fillForm(j.form);await reseed();}}
 if(st.cam_version!==seen.cam_version){{seen.cam_version=st.cam_version;CAM.azim=st.azim;CAM.elev=st.elev;CAM.zoom=st.zoom;if(st.pick){{const j=await (await fetch('/api/scene/info?pick='+encodeURIComponent(st.pick))).json();if(!j.error)showInfo(j);}}render();}}
 if(st.message)$('rstat').textContent=st.message;}}catch(e){{}}finally{{setTimeout(poll,1500);}}}}
poll();
let running=false;
window.runGo=async function(){{playStop();showMovie(null);FRAME=null;if(!specName){{$('runstat').textContent='build a scene first';return;}}const j=await post('/api/scene/run',{{device:$('run_device').value}});if(j.error){{$('runstat').textContent=j.error;return;}}running=true;$('runbtn').disabled=true;$('runstat').textContent=`generate ${{specName}} on ${{j.device}}...`;MOVIE=null;rpoll();}};
window.runStop=async function(){{await post('/api/scene/run',{{stop:true}});}};
// THE RUN IS plexus.pipeline.generate -- the body of Plexus_Main.py -o generate -- on the server's
// VTK thread; its per-frame hook feeds this page's renderer and answers camera moves between
// frames, so orbit and zoom work WHILE the movie is written to graphs_data/studio/<name>/.
// Every movie frame is also kept as a level state, so PLAY replays the run at any camera.
function showMovie(url){{const v=$('movie');if(url){{v.src=url;v.style.display='block';$('view').style.display='none';}}else{{v.pause();v.style.display='none';$('view').style.display='';}}}}
async function rpoll(){{try{{const j=await (await fetch('/api/scene/run')).json();if(j.error&&!j.running){{$('runstat').textContent='error: '+j.error;}}
 else $('runstat').textContent=(j.running?'running: ':(j.stopped?'stopped: ':'done: '))+`frame ${{j.frame}}/${{j.n_frames}}, ${{j.seconds}}s`+(j.ms_per_frame?` (${{j.ms_per_frame.toFixed(0)}} ms/frame, the engine's own clock)`:(j.frame&&j.seconds?` (${{(j.seconds/j.frame*1000).toFixed(0)}} ms/frame incl. the movie)`:''));
 if(j.counts&&j.counts.sets)$('runcounts').textContent=Object.entries(j.counts.sets).filter(([k])=>k!=='half_edge').map(([k,v])=>`${{k}} ${{v}}`).join('  ');
 render();if(j.running){{setTimeout(rpoll,700);}}else{{running=false;$('runbtn').disabled=false;nframes=j.frames_kept||0;$('frame').max=Math.max(nframes-1,0);
  const a=await (await fetch('/api/scene/artefacts?name='+encodeURIComponent(specName))).json();MOVIE=a.mp4||null;
  $('framelab').textContent=(nframes?`${{nframes}} frames kept (every ${{j.keep_every||1}}): PLAY replays at any camera`:'')+(MOVIE?`; MOVIE plays ${{a.dir}}/movie.mp4`:'');}}}}catch(e){{setTimeout(rpoll,1500);}}}}
let MOVIE=null, playing=null, nframes=0, FRAME=null;
async function showFrame(i){{FRAME=i;$('frame').value=i;$('framelab').textContent=`frame ${{i}}/${{Math.max(nframes-1,0)}}`;await render();}}
window.playGo=async function(){{showMovie(null);const j=await (await fetch('/api/scene/frames')).json();nframes=j.n||0;if(!nframes){{$('framelab').textContent='no frames yet: RUN first';return;}}$('frame').max=nframes-1;playing=true;let i=0;
 while(playing){{await showFrame(i);i=(i+1)%nframes;await new Promise(r=>setTimeout(r,30));}}}};
window.playStop=function(){{playing=null;const v=$('movie');if(v.style.display!=='none')v.pause();}};
window.movieGo=function(){{if(!MOVIE){{$('framelab').textContent='no movie yet: RUN first';return;}}playing=null;showMovie(MOVIE);const v=$('movie');v.loop=true;v.play();}};
let cseen=0;
window.claudeGo=async function(){{const t=$('task').value.trim();if(!t)return;$('claude').textContent='';cseen=0;const j=await post('/api/scene/claude',{{task:t,mode:TAB}});if(j.error){{$('cstat').textContent=j.error;return;}}$('cstat').textContent='running...';$('cbtn').disabled=true;}};
window.claudeStop=async function(){{await post('/api/scene/claude',{{stop:true}});}};
window.claudeNew=async function(){{await post('/api/scene/claude',{{new_session:true}});$('claude').textContent+='[new session]\n';}};
async function cpoll(){{try{{const j=await (await fetch('/api/scene/claude?since='+cseen)).json();if(j.lines&&j.lines.length){{const el=$('claude');el.textContent+=j.lines.join('\n')+'\n';el.scrollTop=el.scrollHeight;cseen=j.n;}}
 $('cstat').textContent=j.running?'running... '+j.seconds+'s':(j.error?'error: '+j.error.slice(0,200):(j.n?'done in '+j.seconds+'s':''));$('cbtn').disabled=!!j.running;$('cbtn').classList.toggle('on',!!j.running);}}catch(e){{}}finally{{setTimeout(cpoll,1200);}}}}
cpoll();
// THE SCENE IS THERE WHEN THE PAGE OPENS: a fresh server (or one just reset by a tab switch) has
// no spec, so the default form is built and seeded at once; a server that already holds one is
// picked up by poll() instead.
const q=new URLSearchParams(location.search);
if(q.get('name')){{specName=q.get('name');fetch('/api/scene/spec?name='+encodeURIComponent(specName)+'&tab='+TAB).then(r=>r.json()).then(j=>{{if(j.raw){{$('yamltext').value=j.raw;}}if(j.form)fillForm(j.form);reseed();}});}}
else{{fetch('/api/scene/state').then(r=>r.json()).then(st=>{{if(!st.name)build();}}).catch(()=>{{}});}}
</script></body></html>
"""

PLACEHOLDERS = {
    "bio": "e.g. build a 120-cell cyst with one nucleus per cell and integrins outside, then show me one cell",
    "material": "e.g. drop a snow ball onto a water slab, then run",
    "neurons": "e.g. five assemblies of twelve, sparser cross-talk, then run",
    "metabolism": "e.g. sixty metabolites with more autocatalytic cycles, then run",
}


def page(tab_name: str = "material") -> str:
    import json
    tab = tabs.get(tab_name)
    bar = "".join(f'<a class="{"on" if t == tab_name else ""}" onclick="switchTab(\'{t}\')">{t}</a>' for t in tabs.ORDER)
    return SHELL.format(title=tab.TITLE, css=CSS, tabbar=bar, form=tab.FORM_HTML, form_js=tab.FORM_JS,
                        tab_json=json.dumps(tab_name), pick_dir=tab.PICK_DIR,
                        default_bodies=json.dumps(getattr(tab, "DEFAULT_FORM", {}).get("bodies", [])),
                        placeholder=PLACEHOLDERS.get(tab_name, ""))
