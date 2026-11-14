'use strict';
// Model performance page (Group D): how well the models score on games they never trained on, each
// version side by side, and each change before and after. Data: GET /api/model-performance
// (football_profiler/model_performance.py). Other groups add a section with
// ModelPerformance.addSection({id, title, order, render(data)}) returning HTML. Read-only.
const perfPage={data:null,request:0,sections:[],mounted:false};
const PERF_GREY='#6f8299',PERF_LIME='#c8f55b',PERF_TEAL='#7ddeba',PERF_ORANGE='#ff9f5a';
const PERF_COLOURS={published:PERF_GREY,finetuned:PERF_LIME,tdeed_ablation_partial:PERF_TEAL,tdeed_ablation_hard:PERF_ORANGE};
const PERF_CLASS={Drive:'Carry',DRIVE:'Carry',Pass:'Pass',PASS:'Pass','HIGH PASS':'Lofted pass',Cross:'Cross',CROSS:'Cross',
  'Throw-in':'Throw-in','THROW IN':'Throw-in',Shot:'Shot',SHOT:'Shot',Header:'Header',HEADER:'Header',Tackle:'Tackle',
  'PLAYER SUCCESSFUL TACKLE':'Tackle',Block:'Block','BALL PLAYER BLOCK':'Block',OUT:'Ball out','FREE KICK':'Free kick',GOAL:'Goal',
  pass:'Pass',drive:'Carry',carry:'Carry',high_pass:'Lofted pass',cross:'Cross',throw_in:'Throw-in',shot:'Shot',header:'Header',
  tackle:'Tackle',block:'Block',free_kick:'Free kick',interception:'Interception','throw-in':'Throw-in',clearance:'Clearance'};
const PERF_REASONS={picture:['Credited: ball in his picture box',PERF_LIME],pitch:['Credited: nearest on the pitch',PERF_TEAL],
  touchline:['Credited: touchline rule','#a9cff0'],'no ball':['No player: ball not seen',PERF_ORANGE],
  'ball away from players':['No player: ball away from players','#ffd2a3'],'team not on screen':['No player: team off screen','#5d6f86'],
  'no team':['No player: team unknown','#3a4a5f']};
const perfName=c=>PERF_CLASS[c]||String(c).replaceAll('_',' ');
const perfPct=x=>x==null?'—':Math.round(x*100)+'%';
const perfAp=x=>x==null?'—':Number(x).toFixed(3);

const ModelPerformance={
  addSection(section){perfPage.sections=perfPage.sections.filter(s=>s.id!==section.id).concat(section).sort((a,b)=>(a.order??50)-(b.order??50));renderPerformance();},
  get data(){return perfPage.data;},
};
window.ModelPerformance=ModelPerformance;

function perfMount(){
  if(perfPage.mounted)return;
  const link=document.createElement('link');link.rel='stylesheet';link.href='/static/model_performance.css';document.head.appendChild(link);
  const button=document.createElement('button');button.type='button';button.dataset.tab='performance';button.textContent='Model performance';
  document.querySelector('nav [data-tab="data"]').after(button);
  button.onclick=handler(()=>changeTab('performance'));
  const page=document.createElement('section');page.id='tab-performance';page.className='tab';page.hidden=true;
  page.innerHTML=`<div class="heading-row"><div><p class="eyebrow">Model performance</p><h1>How well the models work</h1>
    <p class="muted">Scored on games the models never trained on, so the numbers are fair. 1.0 or 100% is perfect.</p></div>
    <button type="button" id="perf-reload">Reload scores</button></div><div id="perf-body"><p class="muted">Loading scores…</p></div>`;
  document.querySelector('main').appendChild(page);
  $('perf-reload').onclick=handler(()=>loadPerformance(true));
  perfPage.mounted=true;
}

async function loadPerformance(force=false){
  if(perfPage.data&&!force){renderPerformance();return;}
  const request=++perfPage.request;
  try{const data=await api('/api/model-performance');if(request!==perfPage.request)return;perfPage.data=data;}
  catch(error){if(request===perfPage.request)$('perf-body').innerHTML=`<p class="empty">Could not load the scores: ${esc(error.message)}</p>`;return;}
  renderPerformance();
}

// One labelled bar: value 0-1 (or out of max), drawn in colour.
function perfBar(label,value,colour,{max=1,text=null,note=''}={}){
  const w=value==null?0:Math.max(0,Math.min(100,value/max*100));
  return `<div class="perf-bar"><span class="perf-bar-label">${esc(label)}</span><span class="perf-track"><i style="width:${w}%;background:${colour}"></i></span><strong>${text??(value==null?'waiting':perfAp(value))}</strong>${note?`<small>${esc(note)}</small>`:''}</div>`;
}
function perfCard(title,intro,body,cls=''){return `<article class="perf-card ${cls}"><h2>${esc(title)}</h2>${intro?`<p class="small muted">${intro}</p>`:''}${body}</article>`;}
function perfLegend(items){return '<div class="perf-legend">'+items.map(([name,colour])=>`<span><i style="background:${colour}"></i>${esc(name)}</span>`).join('')+'</div>';}

function perfHeadline(d){
  const h=d.headline||{},tiles=[];
  if(h.spotter_map)tiles.push(['Video model score',perfAp(h.spotter_map.value),h.spotter_map.from!=null?`up from ${perfAp(h.spotter_map.from)} before our training (FOOTPASS, mean AP)`:'FOOTPASS, mean AP']);
  if(h.pass_found!=null)tiles.push(['Passes found',perfPct(h.pass_found),'of the real passes, within 1 s']);
  if(h.pass_right_player!=null)tiles.push(['Right player named',perfPct(h.pass_right_player),h.pass_right_track!=null?`${perfPct(h.pass_right_track)} with perfect names`:'of the real passes']);
  if(h.pass_count_correlation!=null)tiles.push(['Pass counts vs truth','r = '+Number(h.pass_count_correlation).toFixed(2),'per player, with perfect positions']);
  if(h.own_labels_found!=null)tiles.push(['Our own labelled clip',perfPct(h.own_labels_found),'of the actions we labelled were found']);
  if(!tiles.length)return '';
  return '<div class="perf-tiles">'+tiles.map(([k,v,s])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong><small>${esc(s)}</small></div>`).join('')+'</div>';
}

function perfVersions(d){
  const sets=d.spotter?.sets||{},versions=d.spotter?.versions||[];
  if(!Object.keys(sets).length)return '';
  const body=Object.values(sets).map(s=>`<h3>${esc(s.name)}</h3>`+versions.map(v=>perfBar(v.name,s.mean_average_precision[v.tag],PERF_COLOURS[v.tag]||PERF_LIME,
    {note:s.mean_average_precision[v.tag]==null?(v.tag.startsWith('tdeed_ablation')?'overnight run not done yet':'not measured'):''})).join('')).join('');
  return perfCard('Video model versions, side by side','Mean average precision (mAP) over all action types. The two "Retrained" bars differ only in our training trick, so they show whether the trick itself helped.',body);
}

function perfClasses(d){
  const s=d.spotter?.sets?.footpass_val||Object.values(d.spotter?.sets||{})[0];
  if(!s||!Object.keys(s.average_precision||{}).length)return '';
  const rows=Object.entries(s.average_precision).sort((a,b)=>(b[1].finetuned??0)-(a[1].finetuned??0));
  const body=perfLegend([['Public model',PERF_GREY],['Ours',PERF_LIME]])+rows.map(([c,v])=>`<div class="perf-pair"><span class="perf-bar-label">${esc(perfName(c))} <small>${v.labels} labels</small></span><div>`+
    perfBar('Public',v.published,PERF_GREY)+perfBar('Ours',v.finetuned,PERF_LIME)+'</div></div>').join('');
  return perfCard('Each action, before and after our training',`${esc(s.name)}. Average precision per action type. Rare actions (tackles, blocks) stay weak: the public model learned from very few of them.`,body);
}

function perfCrediting(d){
  const cur=d.crediting?.current?.actions,line=d.crediting?.throwinline?.actions;
  if(!cur)return '';
  const order=['Pass','Drive','Shot','Cross','Header','Throw-in','Tackle','Block'];
  const rows=Object.entries(cur).sort((a,b)=>order.indexOf(a[0])-order.indexOf(b[0]));
  let body=perfLegend([['Found',PERF_GREY],['Right player, if names were perfect',PERF_TEAL],['Right named player',PERF_LIME]])+
    rows.map(([c,a])=>`<div class="perf-pair"><span class="perf-bar-label">${esc(perfName(c))} <small>${a.labels} labels</small></span><div>`+
      perfBar('Found',a.found,PERF_GREY,{text:perfPct(a.found)})+perfBar('Right track',a.credited_to_right_track,PERF_TEAL,{text:perfPct(a.credited_to_right_track)})+
      perfBar('Right player',a.credited_correctly,PERF_LIME,{text:perfPct(a.credited_correctly)})+'</div></div>').join('');
  const a=cur['Throw-in'],b=line?.['Throw-in'];
  if(a&&b)body+='<h3>Throw-ins: before and after the touchline rule (C4b)</h3>'+
    perfBar('Before',a.credited_correctly,PERF_GREY,{text:perfPct(a.credited_correctly)})+perfBar('After',b.credited_correctly,PERF_LIME,{text:perfPct(b.credited_correctly)});
  return perfCard('From spotting an action to naming the player','FOOTPASS held-out games, whole pipeline. "Found" is within 1 second. Naming needs both the right track (Group D) and the right name on it (Group C), so the drop between the bars shows where the work is.',body);
}

function perfReasons(d){
  const r=d.reasons?.rule_off;
  if(!r||!Object.keys(r).length)return '';
  const order=['picture','pitch','touchline','no ball','ball away from players','team not on screen','no team'];
  const keys=[...order,...(d.reasons.reasons||[]).filter(k=>!order.includes(k))];
  const body=perfLegend(keys.filter(k=>PERF_REASONS[k]).map(k=>PERF_REASONS[k]))+Object.entries(r).sort((a,b)=>b[1].spots-a[1].spots).map(([kind,row])=>
    `<div class="perf-stack-row"><span class="perf-bar-label">${esc(perfName(kind))} <small>${row.spots}</small></span><span class="perf-stack">`+
    keys.filter(k=>row[k]>0&&PERF_REASONS[k]).map(k=>`<i style="width:${row[k]*100}%;background:${PERF_REASONS[k][1]}" title="${esc(PERF_REASONS[k][0])}: ${perfPct(row[k])}"></i>`).join('')+
    `</span><strong>${perfPct(row.credited)}</strong></div>`).join('');
  return perfCard('Why some spotted actions get no player',`Every confident spot, by how its player was picked or why none was (C4a), over ${d.reasons.datasets||'the'} analysed halves. The number is the share credited.`,body);
}

function perfSpeed(d){
  const t=d.speed?.timing?.settings;
  if(!t)return '';
  const names={before:'Before (take turns)',read_ahead:'Read ahead (C5a, on)',read_ahead_half_overlap:'Read ahead + half overlap (C5b)'};
  const max=Math.max(...Object.values(t).map(r=>r.minutes_per_half||0),1);
  const body=Object.entries(t).map(([k,r])=>perfBar(names[k]||k,r.minutes_per_half,k==='before'?PERF_GREY:PERF_LIME,{max,text:`${Number(r.minutes_per_half).toFixed(1)} min`,note:k==='before'?'':`×${r.speed_up}`})).join('');
  return perfCard('Video model speed','Minutes for the video model to read one 45-minute half on the RTX 3090. Shorter is better.',body);
}

function perfOwnLabels(d){
  const o=d.own_labels;
  if(!o)return perfCard('Against our own labels','Our own labelled clip is not scored yet. Label the clip in the sheet, then run <code>scripts/own_labels.py check</code>.','','perf-waiting');
  const rows=Object.entries(o.actions||{});
  const body=perfLegend([['Found',PERF_GREY],['Right team',PERF_TEAL],['Right player',PERF_LIME]])+
    [['All actions',o.overall],...rows.map(([k,v])=>[perfName(k),v])].map(([name,a])=>`<div class="perf-pair"><span class="perf-bar-label">${esc(name)} <small>${a.labels} labels</small></span><div>`+
      perfBar('Found',a.found,PERF_GREY,{text:perfPct(a.found),note:a.precision!=null?`precision ${perfPct(a.precision)}`:''})+
      perfBar('Right team',a.team_right_when_found,PERF_TEAL,{text:perfPct(a.team_right_when_found)})+
      perfBar('Right player',a.right_player,PERF_LIME,{text:perfPct(a.right_player),note:a.labels_with_jersey?`of ${a.labels_with_jersey} with a readable number`:''})+'</div></div>').join('');
  return perfCard('Against our own labels',`${esc(o.title||o.dataset)}: ${o.overall.labels} actions we labelled ourselves, matched within ${o.tolerance_s} s. "Right team" and "right player" are shares of the ones found.`,body);
}

function perfChanges(d){
  const pill={done:'Done',measured:'Measured',waiting:'Waiting'};
  const rows=(d.changes||[]).map(c=>`<tr><td><strong>${esc(c.id)}</strong></td><td>${esc(c.name)}<br><small class="muted">${esc(c.kind)}</small></td><td>${esc(c.before)}</td><td>${esc(c.after)}</td><td>${esc(c.result)}${c.verdict?`<br><b class="perf-verdict">${esc(c.verdict)}</b>`:''}</td><td><span class="perf-pill ${esc(c.state)}">${esc(pill[c.state]||c.state)}</span></td></tr>`).join('');
  return perfCard('What Group D changed, before and after','Switches stay off until the measured result says they help.',
    `<div class="table-wrap"><table class="perf-table"><thead><tr><th></th><th>Change</th><th>Before</th><th>After</th><th>Measured</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`,'perf-wide');
}

function renderPerformance(){
  if(!perfPage.mounted||!perfPage.data)return;
  const d=perfPage.data;
  const sections=perfPage.sections.map(s=>{try{return s.render(d)||'';}catch(error){return perfCard(s.title||s.id,'This section could not draw: '+esc(error.message),'');}}).join('');
  const missing=d.missing?.length?`<p class="small muted perf-missing">Not measured on this machine yet: ${esc(d.missing.join(', '))}. Results appear here once these runs are done.</p>`:'';
  $('perf-body').innerHTML=perfHeadline(d)+`<div class="perf-grid">${sections}</div>`+missing;
}

for(const [id,order,render] of [['versions',10,perfVersions],['classes',20,perfClasses],['crediting',30,perfCrediting],['reasons',40,perfReasons],
  ['own-labels',50,perfOwnLabels],['speed',60,perfSpeed],['changes',90,perfChanges]])ModelPerformance.addSection({id,title:id,order,render});
perfMount();
window.addEventListener('tabchange',handler(async()=>{if(state.tab==='performance')await loadPerformance();}));
