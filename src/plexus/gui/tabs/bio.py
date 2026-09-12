"""The bio tab: a tissue -- an apico-basal sheet, disc or spheroid of cells -- with its protein
species and organelles, as a form over `gui/bio.py`'s spec builder (the `spheroid_proteins`
template's mechanics, growth and division; the form decides geometry, polarity and the species).
"""
from __future__ import annotations

import os

from plexus.gui import bio, studio

NAME, TITLE = "bio", "Plexus bio objects"
PICK_DIR = os.path.join(studio.REPO, "config", "tissue")
CORPUS_MODE = "bio"
BRIEF = bio.BIO_BRIEF.replace("/api/bio/build", "/api/tab/bio/build").replace("/api/bio/", "/api/scene/")
# THE TAB OPENS A FILE, NOT THE FORM. `ms4_round_prism_shell` is a run's own spec (copied from
# graphs_data/tissue/ms4_round_prism_shell/spec.yaml): a 200-cell apico-basal shell with the
# growth, mechanics, division and edge-flip a tissue run actually uses, which the form -- whose
# template is `spheroid_proteins` -- cannot write. BUILD + SEED still writes the form's spec.
DEFAULT_SPEC = os.path.join(studio.REPO, "config", "tissue", "ms4_round_prism_shell.yaml")
DEFAULT_FORM = {"name": "bio_scene", "shape": "sphere", "n_cells": 200, "radius": 5.0, "h0": 1.2, "apical": "in",
                "world": 50.0, "n_frames": 801,
                "species": [{"name": "integrin", "region": "basal", "density": 3.0, "s": 0.02, "tau": 300.0}],
                "organelles": [{"name": "nucleus", "count": 1, "radius": 0.3, "region": "basal_side",
                                "on_divide": "duplicate", "tau": None}]}


def build_spec(form: dict) -> dict:
    return bio.normalise(bio.build_spec(form))


normalise = bio.normalise
form_from_spec = bio.form_from_spec


FORM_HTML = r'''

 <div class="row"><label>name</label><input id="name" value="bio_scene"></div>
 <div class="row"><label>shape</label><select id="shape"><option>sphere</option><option>disc</option><option>plane</option></select></div>
 <div class="row"><label>cells</label><input id="n_cells" class="short" value="200"> <label style="width:60px">radius</label><input id="radius" class="short" value="5.0"></div>
 <div class="row"><label>thickness h0</label><input id="h0" class="short" value="1.2"> <label style="width:60px">apical</label><select id="apical" style="width:64px"><option value="in">in</option><option value="out">out</option></select></div>
 <div class="row"><label>box</label><input id="world" class="short" value="50"> <label style="width:60px">frames</label><input id="n_frames" class="short" value="801"></div>
 <div class="row"><button class="dim" onclick="addOrganelle()">+ organelle</button> <span style="color:#9ab">organelles</span></div>
 <table class="sp" id="organelles"><tr><th>name</th><th>per cell</th><th>radius</th><th>region</th><th>on divide</th><th>tau</th><th></th></tr></table>
 <div class="row"><button class="dim" onclick="addSpecies()">+ species</button> <span style="color:#9ab">protein species</span></div>
 <div class="row" id="curves" style="color:#ccd">plots: <label style="width:auto;margin-right:8px"><input type="checkbox" value="cells" style="width:auto" onchange="setCurves()"> cells</label><label style="width:auto;margin-right:8px"><input type="checkbox" value="area" style="width:auto" onchange="setCurves()"> area</label><label style="width:auto;margin-right:8px"><input type="checkbox" value="volume" style="width:auto" onchange="setCurves()"> volume</label><label style="width:auto;margin-right:8px"><input type="checkbox" value="radius" style="width:auto" onchange="setCurves()"> radius</label><label style="width:auto;margin-right:8px"><input type="checkbox" value="phase" style="width:auto" onchange="setCurves()"> phase</label><label style="width:auto"><input type="checkbox" value="cycle_progress" style="width:auto" onchange="setCurves()"> cycle</label></div>
 <div style="color:#778;font-size:11px">up to three plots are drawn beside the tissue, live, and in the movie</div>
 <table class="sp" id="species"><tr><th>name</th><th>region</th><th>density</th><th>s</th><th>tau</th><th></th></tr></table>
'''

FORM_JS = r'''
window.addSpecies=function(sp){sp=sp||{};const tb=$('species');const tr=tb.insertRow(-1);
 tr.innerHTML=`<td><input value="${sp.name||''}"></td><td><select><option>basal</option><option>apical</option><option>mid</option><option>interior</option></select></td><td><input value="${sp.density??3}"></td><td><input value="${sp.s??0.02}"></td><td><input value="${sp.tau??300}"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[1].firstChild.value=sp.region||'basal';};
window.addOrganelle=function(og){og=og||{};const tb=$('organelles');const tr=tb.insertRow(-1);
 tr.innerHTML=`<td><input value="${og.name||''}"></td><td><input value="${og.count??1}"></td><td><input value="${og.radius??0.3}"></td><td><select><option>interior</option><option>apical_side</option><option>basal_side</option></select></td><td><select><option>duplicate</option><option>halve</option><option>none</option></select></td><td><input value="${og.tau??''}" placeholder="-"></td><td><button class="dim" onclick="this.closest('tr').remove()">x</button></td>`;
 tr.cells[3].firstChild.value=og.region||'interior';tr.cells[4].firstChild.value=og.on_divide||'duplicate';};
function species(){const out=[];for(const tr of $('species').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 out.push({name,region:c[1].firstChild.value,density:+c[2].firstChild.value,s:+c[3].firstChild.value,tau:+c[4].firstChild.value});}return out;}
function organelles(){const out=[];for(const tr of $('organelles').rows){if(!tr.cells[0].querySelector('input'))continue;const c=tr.cells;const name=c[0].firstChild.value.trim();if(!name)continue;
 const tau=c[5].firstChild.value.trim();out.push({name,count:+c[1].firstChild.value,radius:+c[2].firstChild.value,region:c[3].firstChild.value,on_divide:c[4].firstChild.value,tau:tau?+tau:null});}return out;}
window.tabForm=function(){return {name:$('name').value,shape:$('shape').value,n_cells:+$('n_cells').value,radius:+$('radius').value,h0:+$('h0').value,apical:$('apical').value,world:+$('world').value,n_frames:+$('n_frames').value,species:species(),organelles:organelles()};}
window.tabFill=function(f){$('name').value=f.name;$('shape').value=f.shape;$('n_cells').value=f.n_cells;$('radius').value=f.radius;$('h0').value=f.h0;$('apical').value=f.apical;$('world').value=f.world;$('n_frames').value=f.n_frames;
 let tb=$('species');while(tb.rows.length>1)tb.deleteRow(-1);(f.species||[]).forEach(addSpecies);
 tb=$('organelles');while(tb.rows.length>1)tb.deleteRow(-1);(f.organelles||[]).forEach(addOrganelle);}
// THE PLOT LIST: the quantities `plotting.curve` draws beside the picture; at most three fit.
window.setCurves=async function(){const boxes=[...document.querySelectorAll('#curves input')];const want=boxes.filter(b=>b.checked).map(b=>b.value);
 if(want.length>3){boxes.filter(b=>b.checked).slice(3).forEach(b=>b.checked=false);status('at most three plots fit beside the picture',true);return setCurves();}
 if(!specName){status('build or open a scene first',true);return;}status('drawing '+(want.join(', ')||'no plots')+'...');
 const j=await post('/api/scene/curves',{name:specName,curves:want});if(j.error){status(j.error,true);return;}if(j.version!==undefined)seen.version=j.version;if(j.raw)$('yamltext').value=j.raw;await reseed();};
window.tabInit=function(){addOrganelle({name:'nucleus',count:1,radius:0.3,region:'basal_side',on_divide:'duplicate'});addSpecies({name:'integrin',region:'basal'});};
'''
