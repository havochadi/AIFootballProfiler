'use strict';
const $=id=>document.getElementById(id);
const state={datasets:[],manifest:null,profiles:[],player:null,pid:null,tab:'match',datasetRequest:0,playerRequest:0};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>x==null?'Unavailable':(x*100).toFixed(1)+'%';
const num=x=>x==null?'—':Number(x).toFixed(1);
const base=()=>'/api/datasets/'+encodeURIComponent(state.manifest.id);
const playerBase=()=>base()+'/players/'+encodeURIComponent(state.pid);
let noticeTimer=null;
function clearNotice(){clearTimeout(noticeTimer);$('notice').hidden=true;}
function notice(message,error=false){
  clearTimeout(noticeTimer);const box=$('notice'),text=document.createElement('span'),close=document.createElement('button');
  text.textContent=message;close.type='button';close.textContent='×';close.setAttribute('aria-label','Dismiss notification');close.onclick=clearNotice;
  box.replaceChildren(text,close);box.className=error?'error':'';box.hidden=false;
  if(!error)noticeTimer=setTimeout(clearNotice,10000);
}
async function api(url,options={}){const r=await fetch(url,options);if(!r.ok){let e;try{e=await r.json();}catch{e={detail:r.statusText};}throw Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail));}return r.json();}
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function handler(fn){return async e=>{try{await fn(e);}catch(error){notice(error.message,true);}};}
function table(headers,rows){if(!rows.length)return '<p class="empty">No records available.</p>';return '<div class="table-wrap"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function pitch(canvas,heat=null){const c=canvas.getContext('2d'),W=canvas.width,H=canvas.height,p=16,fw=W-2*p,fh=H-2*p;c.clearRect(0,0,W,H);c.fillStyle='#173e31';c.fillRect(0,0,W,H);if(heat){const max=Math.max(...heat.flat(),1e-8);heat.forEach((row,j)=>row.forEach((v,i)=>{if(v<=0)return;const t=Math.sqrt(v/max);c.fillStyle=`rgba(111,235,149,${.15+.85*t})`;c.fillRect(p+i*fw/32,p+j*fh/20,fw/32+.4,fh/20+.4);}));}c.strokeStyle='rgba(237,255,238,.75)';c.lineWidth=1.6;c.strokeRect(p,p,fw,fh);c.beginPath();c.moveTo(W/2,p);c.lineTo(W/2,H-p);c.stroke();c.beginPath();c.ellipse(W/2,H/2,9.15/105*fw,9.15/68*fh,0,0,Math.PI*2);c.stroke();for(const right of [false,true]){const x=right?W-p-16.5/105*fw:p;c.strokeRect(x,H/2-20.16/68*fh,16.5/105*fw,40.32/68*fh);const gx=right?W-p-5.5/105*fw:p;c.strokeRect(gx,H/2-9.16/68*fh,5.5/105*fw,18.32/68*fh);}return c;}
function safeLocalGet(key){try{return localStorage.getItem(key);}catch{return null;}}
function safeLocalSet(key,value){try{localStorage.setItem(key,value);return true;}catch{return false;}}
async function refreshSources(preferred){
  state.datasets=await api('/api/datasets');
  const full=state.datasets.filter(m=>String(m.analysis||'').startsWith('full-match'));
  const whole=full.filter(m=>m.whole_match),halves=full.filter(m=>!m.whole_match);
  const other=state.datasets.filter(m=>!full.includes(m));
  const options=rows=>rows.map(m=>`<option value="${esc(m.id)}">${esc(m.title)}</option>`).join('');
  // Whole matches are the unit to rate players in; their halves stay available below.
  $('dataset-select').innerHTML=(whole.length?`<optgroup label="Whole matches">${options(whole)}</optgroup>`:'')+
    `<optgroup label="${whole.length?'Single halves':'Analysed matches'}">${options(halves)}</optgroup><optgroup label="Reference data & clips">${options(other)}</optgroup>`;
  const selected=[preferred,safeLocalGet('pitchprofile-match'),whole[0]?.id,full[0]?.id,state.datasets[0]?.id].find(id=>state.datasets.some(m=>m.id===id));
  if(selected)await selectDataset(selected);
  const s=await api('/api/status');$('top-status').textContent=full.length+' analysed halves · '+s.datasets+' sources';
}
function setPlayerControls(enabled){
  for(const id of ['identity-form','event-form'])
    $(id).querySelectorAll('input,select,textarea,button').forEach(control=>control.disabled=!enabled);
}
function clearPlayer(){
  state.pid=null;state.player=null;setPlayerControls(false);window.dispatchEvent(new Event('profilechange'));
  $('player-name').textContent='No tracked people found';
  for(const id of ['player-subtitle','source-kind','direction-note','heatmap-note','zones','event-summary','event-note','manual-events','history','history-note'])$(id).textContent='';
  for(const id of ['position-coverage','detection-coverage','observed-time'])$(id).textContent='—';
  for(const id of ['identity-form','event-form'])$(id).reset();
  $('media-container').innerHTML=state.tab==='overview'?mediaMarkup(true):'';
  $('media-note').textContent=state.manifest.note||'';
  pitch($('heatmap'));
}
async function selectDataset(id,preferredPid=null){
  const request=++state.datasetRequest; ++state.playerRequest;
  window.dispatchEvent(new Event('profilechanging'));
  const previous=state.manifest?.id===id?state.pid:null;
  let data;
  try{data=await api('/api/datasets/'+encodeURIComponent(id));}catch(error){if(request===state.datasetRequest)showEvidenceError('Could not load this match. Retry when the connection is available.');throw error;}
  if(request!==state.datasetRequest)return;
  const changed=state.manifest?.id!==id;
  state.manifest=data.manifest;state.profiles=data.profiles;state.pid=null;
  $('dataset-select').value=id;$('source-note').textContent=data.manifest.note||'';$('player-search').value='';
  $('match-title').textContent=data.manifest.title;$('export-tracks').href=base()+'/export';
  safeLocalSet('pitchprofile-match',id);
  renderPlayers();
  if(state.profiles.length){
    const preferred=state.profiles.find(p=>p.player_id===(preferredPid||previous||safeLocalGet('pitchprofile-player:'+id)))||state.profiles.find(p=>p.player_id==='38673')||state.profiles[0];
    await selectPlayer(preferred.player_id);
  }else{clearPlayer();notice('No people were tracked in this clip. Try a clearer or closer view.',true);}
}
function mediaMarkup(overlay=false){const m=state.manifest;const file=overlay?(m.overlay||m.video):m.video;const prefix='/media/'+encodeURIComponent(m.id)+'/';if(file)return `<video controls preload="metadata" src="${prefix+encodeURIComponent(file)}"></video>`;if(m.preview)return `<img src="${prefix+encodeURIComponent(m.preview)}" alt="First available source frame">`;return '<p class="empty">The source video is not included for this dataset. Review independently sourced footage before drawing conclusions.</p>';}
async function selectPlayer(pid){if(pid===state.pid&&reviewDesk.ready&&currentMatch())return;window.dispatchEvent(new Event('profilechanging'));const request=++state.playerRequest;state.pid=pid;const root=playerBase();let p;try{p=await api(root);}catch(error){if(request===state.playerRequest)showEvidenceError('Could not load this player. Retry loading.');throw error;}if(request!==state.playerRequest)return;state.player=p;safeLocalSet('pitchprofile-player:'+state.manifest.id,pid);setPlayerControls(true);renderPlayers();$('player-name').textContent=p.name;$('player-subtitle').textContent=p.team+' · '+p.role;$('source-kind').textContent={provider_tracking:'Provider tracking',reference_annotations:'Reference annotations',model_predictions:'Model predictions',imported_tracking:'Imported tracking',soccertrack_reference:'SoccerTrack reference'}[p.source_kind]||p.source_kind;$('position-coverage').textContent=pct(p.position_coverage);$('detection-coverage').textContent=pct(p.detected_coverage);$('observed-time').textContent=num(p.observed_seconds/60)+' min';pitch($('heatmap'),p.features_available?p.heatmap:null);$('direction-note').textContent=p.direction_known?'Attack normalised to the right':'Attack direction unconfirmed';$('heatmap-note').textContent=p.features_available?p.heatmap_note+' '+p.coverage_note:'Pitch positions are unavailable. Calibrate uploaded footage before interpreting a heatmap.';$('zones').innerHTML=p.zone_shares.map((v,i)=>`<div>${p.direction_known?['Defensive third','Middle third','Attacking third'][i]:['Left third','Middle third','Right third'][i]}<strong>${pct(v)}</strong></div>`).join('');renderEvents();$('media-container').innerHTML=state.tab==='overview'?mediaMarkup(true):'';$('media-note').textContent=state.manifest.source_kind==='model_predictions'?'Overlay IDs are model outputs. Confirm identity and remove irrelevant people from your interpretation.':state.manifest.note;const mp=p.manifest_player;$('identity-name').value=mp.name;$('identity-team').value=mp.team||'';$('identity-role').value=mp.role||'';$('identity-group').value=mp.position_group||'';$('identity-key').value=mp.global_id||'';$('identity-verified').checked=mp.identity_verified||false;$('identity-direction').value=mp.direction||'unknown';$('identity-direction').disabled=!['model_predictions','soccertrack_reference'].includes(state.manifest.source_kind);const history=await api(root+'/history');if(request!==state.playerRequest)return;$('history').innerHTML=table(['Date','Role','Minutes','Pass-ending possessions /90','Runs behind /90'],history.matches.map(r=>[r.date.slice(0,10),r.position_group,num(r.minutes_played),num(r.possession_end_pass_per90),num(r.run_behind_per90)]));$('history-note').textContent=history.note;window.dispatchEvent(new Event('profilechange'));}
function renderEvents(){const e=state.player.events;if(e){const map={possession_end_pass:'Possessions ending in a pass',possession_end_shot:'Possessions ending in a shot',run_behind:'Runs behind',on_ball_engagement:'On-ball engagements'};$('event-summary').innerHTML='<p class="source-label">'+esc(e.source)+'</p>'+Object.entries(map).map(([k,label])=>`<div class="score"><span>${label}</span><strong>${e.counts[k]??'—'}</strong></div>`).join('');$('event-note').textContent=e.note;}else{$('event-summary').innerHTML='<p class="empty">No complete matched event feed.</p>';$('event-note').textContent='Add individual events below. Missing records are not counted as zero actions.';}$('manual-events').innerHTML=table(['Seconds','Event','Reviewer','Notes'],state.player.manual_events.map(e=>[num(e.time_s),e.kind,e.reviewer,e.notes]));}
async function changeTab(tab){
  state.tab=tab;document.body.dataset.page=tab;
  document.querySelectorAll('nav button[data-tab]').forEach(b=>{const active=b.dataset.tab===tab;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  document.querySelectorAll('main>.tab').forEach(s=>s.hidden=s.id!=='tab-'+tab);
  document.querySelector('.tools-menu').open=false;
  document.querySelectorAll('video').forEach(v=>{if(v.closest('.tab')?.hidden)v.pause();});
  if(tab==='overview'&&state.manifest&&!$('media-container').children.length)$('media-container').innerHTML=mediaMarkup(true);
  if(tab!=='overview')$('media-container').replaceChildren();
  window.dispatchEvent(new Event('tabchange'));
}
async function monitor(jobId,preferred){$('job').hidden=false;while(true){const j=await api('/api/jobs/'+jobId);$('job-message').textContent=j.message;$('job-progress').value=j.progress;$('job-value').textContent=Math.round(j.progress*100)+'%';if(j.status==='failed'){$('job').hidden=true;throw Error(j.message);}if(j.status==='complete'){$('job').hidden=true;notice('Processing complete. Results are saved.');await refreshSources(preferred||state.manifest?.id);return j;}await new Promise(resolve=>setTimeout(resolve,1200));}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=handler(()=>changeTab(b.dataset.tab)));
$('dataset-select').onchange=handler(e=>selectDataset(e.target.value));$('player-search').oninput=()=>renderPlayers();
$('identity-form').onsubmit=handler(async e=>{e.preventDefault();const savedPid=state.pid;await post(playerBase()+'/identity',{name:$('identity-name').value,team:$('identity-team').value,role:$('identity-role').value,position_group:$('identity-group').value||null,global_id:$('identity-key').value||null,identity_verified:$('identity-verified').checked,direction:$('identity-direction').value});notice('Identity saved.');await selectDataset(state.manifest.id,savedPid);});
$('event-form').onsubmit=handler(async e=>{e.preventDefault();const r=await post(playerBase()+'/events',{time_s:Number($('event-time').value),kind:$('event-kind').value,reviewer:$('event-reviewer').value,notes:$('event-notes').value});state.player.manual_events=r.events;renderEvents();notice('Reviewed event saved.');});
$('import-form').onsubmit=handler(async e=>{e.preventDefault();const fd=new FormData();fd.append('tracks',$('import-tracks').files[0]);fd.append('manifest',$('import-manifest').files[0]);const m=await api('/api/import-tracks',{method:'POST',body:fd});notice('Tracking dataset imported.');await refreshSources(m.id);await changeTab('overview');});
$('export-manifest').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(state.manifest,null,2)],{type:'application/json'}));a.download=state.manifest.id+'_manifest.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);};
setPlayerControls(false);pitch($('heatmap'));document.body.dataset.page=state.tab;
window.addEventListener('DOMContentLoaded',()=>refreshSources().catch(e=>notice(e.message,true)));
