'use strict';
let libraryVideos=[];
let analysisRequest=0;
function eventTable(events,clickable=false){
  if(!events.length)return '<p class="empty">No annotations in this interval.</p>';
  return '<div class="table-wrap"><table><thead><tr><th>Seconds</th><th>Event</th><th>Team</th><th>Visibility</th></tr></thead><tbody>'+events.map(e=>`<tr><td>${clickable?`<button type="button" data-event-time="${Number(e.time_s)}">${num(e.time_s)}</button>`:num(e.time_s)}</td><td>${esc(e.label)}</td><td>${esc(e.team)}</td><td>${esc(e.visibility)}</td></tr>`).join('')+'</tbody></table></div>';
}
async function loadLibrary(){
  const previous=$('library-video').value;
  const data=await api('/api/soccernet/library');libraryVideos=data.videos;
  $('library-path').textContent=data.root;
  $('library-video').innerHTML=data.videos.map(v=>`<option value="${esc(v.id)}">${esc(v.title)} · Half ${v.half}</option>`).join('');
  if(data.videos.some(v=>v.id===previous))$('library-video').value=previous;
  $('library-analyse').disabled=!data.videos.length;
  if(!data.videos.length){$('library-info').textContent='No downloaded 720p halves were found.';return;}
  await librarySelection();
}
async function librarySelection(){
  const id=$('library-video').value,v=libraryVideos.find(v=>v.id===id);if(!v)return;
  $('library-info').textContent=`${v.width} × ${v.height} · ${num(v.duration/60)} minutes · official split: ${v.benchmark_split||'unknown'} · ${v.actions_available?'Match labels available':'No match labels downloaded'}. The analysed interval uses a clock starting at zero.`;
  $('library-start').max=Math.max(0,v.duration-1);
  const result=await api('/api/soccernet/library/'+encodeURIComponent(id)+'/events');
  if($('library-video').value!==id)return;
  $('library-events').innerHTML=eventTable(result.events,true);
  $('library-events').querySelectorAll('[data-event-time]').forEach(b=>b.onclick=()=>{$('library-start').value=Math.max(0,Number(b.dataset.eventTime)-10).toFixed(1);notice('Analysis start set to 10 seconds before the event.');});
}
function speedPlot(data){
  const canvas=$('speed-chart'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  c.clearRect(0,0,w,h);c.fillStyle='#152d26';c.fillRect(0,0,w,h);
  const trace=data.speed_trace,values=trace.filter(p=>p.speed_kmh!=null),max=Math.max(25,...values.map(p=>p.speed_kmh));
  c.fillStyle='#ddd';c.font='14px sans-serif';c.fillText('km/h',8,20);c.fillText(max.toFixed(0),8,42);c.fillText('0',22,h-30);
  c.fillText('Seconds in analysed interval',w-210,h-8);
  if(!values.length){c.fillText('Calibrated consecutive positions are required.',65,110);return;}
  const end=Math.max(1,...trace.map(p=>p.time_s));c.fillText(end.toFixed(1),w-65,h-28);
  let drawing=false;c.strokeStyle='#73e8a2';c.lineWidth=2;c.beginPath();
  trace.forEach(p=>{if(p.speed_kmh==null){drawing=false;return;}const x=55+p.time_s/end*(w-80),y=h-45-p.speed_kmh/max*(h-75);if(drawing)c.lineTo(x,y);else c.moveTo(x,y);drawing=true;});c.stroke();
}
async function loadMovement(){
  const request=++analysisRequest;
  if(!state.manifest||!state.pid){$('motion-player').textContent='Choose a player';$('motion-summary').textContent='';return;}
  const currentBase=base();
  const [data,events]=await Promise.all([api(playerBase()+'/analytics'),api(currentBase+'/match-events')]);
  if(request!==analysisRequest||currentBase!==base())return;
  $('motion-player').textContent=state.player.name;
  const m=data.motion;
  $('motion-summary').innerHTML=table(['Observed distance (m)','Valid motion (s)','Mean speed (km/h)','Peak speed (km/h)','Peak |acceleration| (m/s²)','Excluded steps'],[[num(m.distance_m),num(m.observed_motion_seconds),num(m.mean_speed_kmh),num(m.peak_speed_kmh),num(m.peak_abs_acceleration_ms2),m.excluded_steps]]);
  $('motion-zones').innerHTML=table(['Intensity','Range (km/h)','Observed seconds'],m.zones.map(z=>[z.id.replaceAll('_',' '),z.lower_kmh+' – '+(z.upper_kmh??'above'),num(z.seconds)]));
  $('motion-note').textContent=m.note;speedPlot(m);
  const kit=data.kit_suggestion;
  $('kit-suggestion').textContent=kit?`Kit suggestion: ${kit.group} · ${pct(kit.vote_share)} of ${kit.samples} colour observations. ${data.kit_note}`:'No kit-colour suggestion for this track.';
  $('match-events').innerHTML=eventTable(events.events);$('match-events-note').textContent=events.note;
  $('export-mot').href=currentBase+'/mot';$('export-mot').hidden=state.manifest.source_kind!=='model_predictions';
  $('render-overlay').hidden=state.manifest.source_kind!=='model_predictions';
  $('tactics-time').max=Math.max(0,state.manifest.duration_seconds-.1);
  if(Number($('tactics-time').value)>=state.manifest.duration_seconds)$('tactics-time').value=0;
  await loadTactics();
}
let tacticsRequest=0;
async function loadTactics(){
  if(!state.manifest)return;
  const request=++tacticsRequest,root=base();
  const data=await api(root+'/tactics?seconds='+Number($('tactics-time').value));
  if(request!==tacticsRequest||root!==base())return;
  const c=pitch($('tactics-pitch')),point=p=>[16+p[0]/105*808,16+p[1]/68*512];
  const teams=[...new Set(data.players.map(p=>p.team))].sort(),palette=['#f4ca56','#60c3f8','#e986c5','#dbe9de'];
  const colour=t=>palette[Math.max(0,teams.indexOf(t))%palette.length];
  data.teams.forEach(t=>{if(!t.hull.length)return;c.beginPath();t.hull.forEach((p,i)=>{const [x,y]=point(p);if(i)c.lineTo(x,y);else c.moveTo(x,y);});c.closePath();c.fillStyle=colour(t.team)+'30';c.strokeStyle=colour(t.team);c.fill();c.stroke();});
  data.players.forEach(p=>{const [x,y]=point([p.x,p.y]);c.beginPath();c.arc(x,y,p.player_id===state.pid?7:5,0,Math.PI*2);c.fillStyle=colour(p.team);c.fill();if(p.player_id===state.pid){c.strokeStyle='#fff';c.lineWidth=2;c.stroke();}});
  $('tactics-teams').innerHTML=table(['Team','Visible outfield','Width (m)','Length (m)','Hull area (m²)','Line estimate*'],data.teams.map(t=>[t.team,t.visible_outfield,num(t.width_m),num(t.length_m),num(t.hull_m2),t.line_estimate||'Unavailable']));
  $('tactics-note').textContent=(data.time_s==null?'':`Frame at ${num(data.time_s)}s. `)+data.note+' *Line estimate: ten identified outfield players grouped by gaps over 6 m along the pitch; an instantaneous heuristic, not a verified formation.';
}
async function loadOccupancy(){
  const root=base();$('load-occupancy').disabled=true;
  try{
    const data=await api(root+'/occupancy');if(root!==base())return;
    $('occupancy-note').textContent=data.note;
    $('occupancy-maps').innerHTML=data.teams.map((t,i)=>`<div><h3>${esc(t.team)} · ${t.observations} observations</h3><canvas id="occupancy-${i}" width="840" height="544" aria-label="${esc(t.team)} observed occupancy"></canvas></div>`).join('');
    data.teams.forEach((t,i)=>pitch($('occupancy-'+i),t.heatmap));
    if(data.difference){
      const d=data.difference;$('occupancy-maps').insertAdjacentHTML('beforeend',`<div><h3>Presence difference</h3><p>Gold: ${esc(d.positive_team)} · Blue: ${esc(d.negative_team)}</p><canvas id="occupancy-diff" width="840" height="544" aria-label="Difference between observed team occupancy"></canvas></div>`);
      const c=pitch($('occupancy-diff')),max=Math.max(...d.heatmap.flat().map(Math.abs),1e-8);
      d.heatmap.forEach((row,j)=>row.forEach((v,i)=>{c.fillStyle=v>=0?`rgba(244,202,86,${.75*Math.sqrt(v/max)})`:`rgba(96,195,248,${.75*Math.sqrt(-v/max)})`;c.fillRect(16+i*808/32,16+j*512/20,808/32,512/20);}));
    }
    if(!data.teams.length)$('occupancy-maps').textContent='Confirmed teams and common pitch coordinates are required.';
  }finally{$('load-occupancy').disabled=false;}
}
$('load-occupancy').onclick=handler(loadOccupancy);
$('render-overlay').onclick=handler(async()=>{const id=state.manifest.id;const result=await post(base()+'/render-overlay',{});await monitor(result.job_id,id);await changeTab('overview');});
window.addEventListener('profilechange',()=>{$('occupancy-maps').textContent='';$('occupancy-note').textContent='';});
$('library-video').onchange=handler(librarySelection);
$('library-form').onsubmit=handler(async e=>{e.preventDefault();const result=await post('/api/soccernet/analyse',{library_id:$('library-video').value,start_s:Number($('library-start').value),max_seconds:Number($('library-duration').value),sampling_hz:Number($('library-hz').value),tracker:$('library-tracker').value});await monitor(result.job_id,result.dataset_id);await changeTab('overview');});
$('tactics-form').onsubmit=handler(async e=>{e.preventDefault();await loadTactics();});
window.addEventListener('tabchange',handler(async()=>{if(state.tab==='data')await loadLibrary();if(state.tab==='analysis')await loadMovement();}));
window.addEventListener('profilechange',handler(async()=>{if(state.tab==='analysis')await loadMovement();}));
