'use strict';
const matchState={data:null,request:0,playerRequest:0,sort:'visible_seconds',team:'',basis:'per90_visible'};
const isMatch=()=>String(state.manifest?.analysis||'').startsWith('full-match');
const matchBase=()=>'/api/datasets/'+encodeURIComponent(state.manifest.id);
const perStat=(p,k)=>p[matchState.basis]?.[k]??null;
const RATE_SUFFIX={per90_visible:'/90 on screen',per90:'/90 team time',per100_touches:'/100 touches'};
const rateSuffix=()=>RATE_SUFFIX[matchState.basis];
const clock=s=>s==null?'—':Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');
const SORT_VALUE={visible_seconds:p=>p.visible_seconds,distance_per_min_m:p=>p.physical?.distance_per_min_m,top_speed_kmh:p=>p.physical?.top_speed_kmh};
function sortValue(p,key){return (SORT_VALUE[key]||(x=>perStat(x,key)))(p)??-1;}
const evidence=e=>e.evidence||'';

async function loadMatch(){
  const request=++matchState.request,id=state.manifest?.id;
  if(!state.manifest||!isMatch()){matchState.data=null;$('match-content').hidden=true;$('match-empty').hidden=false;$('match-heading').textContent='Choose an analysed match';return;}
  let data;
  try{data=await api(matchBase()+'/match');}catch(error){if(request===matchState.request)showEvidenceError('Could not load match evidence. Retry loading.');throw error;}
  if(request!==matchState.request||id!==state.manifest?.id)return;
  matchState.data=data;$('match-empty').hidden=true;$('match-content').hidden=false;
  renderMatchContext();
  $('match-heading').textContent=data.title;
  for(const k of ['A','B']){const t=data.teams?.[k]||{};$('team-'+k.toLowerCase()+'-name').value=t.name||('Team '+k);$('team-'+k.toLowerCase()+'-swatch').style.background=t.colour||'#ccc';}
  $('match-team-filter').options[1].textContent=data.teams?.A?.name||'Kit group A';$('match-team-filter').options[2].textContent=data.teams?.B?.name||'Kit group B';
  const c=data.coverage||{},fx=data.fixture_teams||{};
  $('match-coverage').textContent=`Live calibrated view ${num((c.live_seconds||0)/60)} min (${pct(c.pitch_view_share)} of the half) · ball seen in ${pct(c.ball_observed_share)} of live frames · attack: ${data.teams?.A?.name} → ${data.teams?.A?.attacks}, ${data.teams?.B?.name} → ${data.teams?.B?.attacks}.`+(fx.home?` Fixture: ${fx.home} v ${fx.away}; name the kit groups to match.`:'');
  $('team-table').innerHTML=table(['Team','Possession share','Passes','Completed','Progressive','Shots','Tackles','Interceptions','Recoveries'],(data.team_stats||[]).map(t=>{const o=t.on_ball||{};return [data.teams?.[t.team]?.name||t.team,pct(t.possession_share),o.passes??'—',o.passes_completed??'—',o.progressive_passes??'—',o.shots??'—',o.tackles??'—',o.interceptions??'—',o.recoveries??'—'];}));
  $('match-note').textContent=data.note||'';
  renderMatchPlayers();renderPlayers();
  if(typeof renderNaming==='function')renderNaming();
  await showMatchPlayer();
}

function renderMatchPlayers(){
  const data=matchState.data;if(!data)return;
  const rows=data.players.filter(p=>!matchState.team||p.team===matchState.team).sort((a,b)=>sortValue(b,matchState.sort)-sortValue(a,matchState.sort));
  const unit=matchState.basis==='per100_touches'?' /100 t':' /90';
  const head=['Player','Group','Visible','Pass'+unit,'Pass %','Prog.'+unit,'Carry'+unit,'Dribble'+unit,'Shot'+unit,'Tackle'+unit,'Int.'+unit,'Recov.'+unit,'Press.'+unit,'m/min','Top km/h'];
  $('match-players').innerHTML='<div class="table-wrap"><table class="clickable"><thead><tr>'+head.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(p=>{
    const o=p.on_ball||{},ph=p.physical||{};
    return `<tr data-player="${esc(p.identity)}" class="${p.identity===state.pid?'selected':''}"><td><strong>${esc(p.name)}</strong></td><td>${esc((p.position_group||'—').replaceAll('_',' '))}</td><td>${clock(p.visible_seconds)}</td><td>${num(perStat(p,'passes'))}</td><td>${o.pass_completion==null?'—':Math.round(o.pass_completion*100)+'%'}</td><td>${num(perStat(p,'progressive_passes'))}</td><td>${num(perStat(p,'carries'))}</td><td>${num(perStat(p,'dribbles'))}</td><td>${num(perStat(p,'shots'))}</td><td>${num(perStat(p,'tackles'))}</td><td>${num(perStat(p,'interceptions'))}</td><td>${num(perStat(p,'recoveries'))}</td><td>${num(perStat(p,'pressures'))}</td><td>${num(ph.distance_per_min_m)}</td><td>${num(ph.top_speed_kmh)}</td></tr>`;
  }).join('')+'</tbody></table></div>';
  $('match-players').querySelectorAll('tr[data-player]').forEach(r=>r.onclick=handler(()=>selectPlayer(r.dataset.player)));
  $('mp-identities').innerHTML=data.players.map(p=>`<option value="${esc(p.identity)}">${esc(p.name)}</option>`).join('');
}

function statList(rows){return '<div class="stat-list">'+rows.map(([label,value,sub])=>`<div><span>${esc(label)}</span><strong>${value}</strong>${sub?`<small>${esc(sub)}</small>`:''}</div>`).join('')+'</div>';}
function playerStats(p,movementOnly=false){
  const o=p.on_ball||{},ph=p.physical||{},pos=p.positional||{},n=k=>o[k]??0,per=k=>perStat(p,k)==null?'':num(perStat(p,k))+' '+rateSuffix();
  const actions='<h4>On the ball</h4>'+statList([
    ['Touches',n('touches'),per('touches')],['Time on ball',num(o.time_on_ball_s)+' s',''],
    ['Passes',`${n('passes_completed')}/${n('passes')}`,o.pass_completion==null?'':Math.round(o.pass_completion*100)+'% completed'],
    ['Progressive passes',n('progressive_passes'),per('progressive_passes')],['Long passes',n('long_passes'),''],['Crosses',n('crosses'),''],['Key passes',n('key_passes'),''],
    ['Mean pass length',o.mean_pass_length_m==null?'—':num(o.mean_pass_length_m)+' m',o.forward_pass_share==null?'':Math.round(o.forward_pass_share*100)+'% forward'],
    ['Carries',n('carries'),n('progressive_carries')+' progressive'],['Dribbles past opponents',n('dribbles'),per('dribbles')],
    ['Shots',n('shots'),o.mean_shot_distance_m==null?'':num(o.mean_shot_distance_m)+' m mean distance']])+
  '<h4>Without the ball</h4>'+statList([['Tackles',n('tackles'),per('tackles')],['Interceptions',n('interceptions'),per('interceptions')],
    ['Recoveries',n('recoveries'),per('recoveries')],['Clearances',n('clearances'),''],['Pressures',n('pressures'),per('pressures')],['Dispossessed',n('dispossessed'),'']]);
  return (movementOnly?'':actions)+'<h4>Physical (visible time only)</h4>'+statList([['Visible',clock(p.visible_seconds),p.segments+' segments'],['Distance',num((ph.distance_m||0)/1000)+' km',num(ph.distance_per_min_m)+' m/min'],
    ['Top speed',ph.top_speed_kmh==null?'—':num(ph.top_speed_kmh)+' km/h','98th percentile'],['Sprints',ph.sprints??0,'≥25.2 km/h for 1 s']])+
  '<h4>Position (attacking right)</h4>'+statList([['Average position',pos.mean_x==null?'—':`${num(pos.mean_x)} m, ${num(pos.mean_y)} m`,'x from own goal line'],
    ...(!movementOnly?[['Thirds',`${pct(pos.defensive_third)} / ${pct(pos.middle_third)} / ${pct(pos.attacking_third)}`,'defensive / middle / attacking']]:[]),
    ['Lanes',`${pct(pos.left_lane)} / ${pct(pos.central_lane)} / ${pct(pos.right_lane)}`,'left / central / right'],['In the penalty box',pct(pos.box_share),'']]);
}
const PROFILE_SHARES=['pass_completion','take_on_success','between_lines_share'];
function styleProfile(p){
  const pr=p.profile;
  if(!pr)return '<p class="empty">Percentile profiles cover outfield players only.</p>';
  const value=s=>s.value==null?'—':PROFILE_SHARES.includes(s.key)?pct(s.value):num(s.value);
  const peers=pr.peer_basis==='same position group'?`${pr.peers} ${String(pr.peer_group).replaceAll('_',' ')} appearances`:`${pr.peers} outfield appearances (too few in this position group)`;
  return `<p class="small muted">Percentile against ${esc(peers)}: the share of them with a lower value. Rates are per 90 minutes of identified screen time; tackles, blocks, headers, crosses and lofted passes are expected counts from the video model.${pr.eligible?'':' This player was identified for under 10 minutes, so treat the ranks with caution.'}</p>`+
    pr.sections.map(sec=>`<h4>${esc(sec.name)}</h4><div class="profile-bars">`+sec.stats.map(s=>{
      const pc=s.percentile==null?null:Math.round(s.percentile);
      return `<div class="profile-row"><span>${esc(s.label)}${s.lower_better?' <small>lower is better</small>':''}</span><strong>${value(s)}</strong><div class="bar" role="img" aria-label="${pc==null?'Not ranked':pc+'th percentile'}"><i style="width:${pc??0}%"></i></div><em>${pc??'—'}</em></div>`;}).join('')+'</div>').join('');
}
function eventStatSummary(p,kind){
  const o=p.on_ball||{},key={pass:'passes',shot:'shots',tackle:'tackles',interception:'interceptions',carry:'carries',dribble:'dribbles',take_on:'take_ons',recovery:'recoveries',clearance:'clearances',pressure:'pressures',touch:'touches',block:'blocks',header:'headers',high_pass:'lofted_passes',cross:'crosses'}[kind];
  const rows=[[eventNames[kind]+' · whole half',o[key]??'—',clock(p.visible_seconds)+' on screen'],['Rate',num(perStat(p,key)),rateSuffix()]];
  if(kind==='pass')rows.push(['Completed',o.passes_completed??'—',pct(o.pass_completion)],['Progressive',o.progressive_passes??'—',''],['Mean length',o.mean_pass_length_m==null?'—':num(o.mean_pass_length_m)+' m','']);
  if(kind==='shot')rows.push(['Mean distance',o.mean_shot_distance_m==null?'—':num(o.mean_shot_distance_m)+' m','Outcomes below need review']);
  if(kind==='carry')rows.push(['Progressive',o.progressive_carries??'—','']);
  if(kind==='dribble')rows.push(['Dispossessed',o.dispossessed??'—','']);
  if(kind==='take_on')rows.push(['Kept the ball',o.dribbles??'—',o.take_on_success==null?'':pct(o.take_on_success)+' of attempts']);
  if(['tackle','block','header','high_pass','cross'].includes(kind)&&p.spotter)rows.push(['Source','Video action spotter','Credited to the player at the ball']);
  if(kind==='touch')rows.push(['Time on ball',num(o.time_on_ball_s)+' s','']);
  const detail=kind==='pass'?'<details class="compact-details"><summary>More passing statistics</summary>'+statList([['Long passes',o.long_passes??'—',''],['Crosses',o.crosses??'—',''],['Key passes',o.key_passes??'—',''],['Forward share',pct(o.forward_pass_share),'']])+'</details>':'';
  return statList(rows)+detail+'<p class="small muted">Whole-half model statistics. The map and outcome counts below follow your time and outcome filters.</p>';
}

function seekTable(headers,rows){
  if(!rows.length)return '<p class="empty">No records available.</p>';
  const html='<div class="table-wrap"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(([t,...rest])=>`<tr><td><button type="button" class="seek" data-seek="${Number(t)}">▶ ${clock(t)}</button></td>`+rest.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
  return html;
}
function loadPlayerVideo(boxes){
  const video=$('mp-video'),m=state.manifest;
  const src=m.video?'/media/'+encodeURIComponent(m.id)+'/'+encodeURIComponent(m.video):'';
  if(video.dataset.src!==src){
    video.pause();delete video.dataset.seek;video.dataset.src=src;
    if(src)video.src=src;else video.removeAttribute('src');
    video.load();
  }
  video.parentElement.hidden=!src;$('mp-video-note').hidden=false;
  $('mp-video-recovery').hidden=!src;$('mp-watch').disabled=!src;
  if(src)$('mp-video-open').href=src;else $('mp-video-open').removeAttribute('href');
  updateVideoStatus();
  matchState.boxes=boxes.boxes;matchState.boxTimes=boxes.boxes.map(b=>b[0]);
  const canvas=$('mp-video-overlay');canvas.width=boxes.width||1280;canvas.height=boxes.height||720;
  drawPlayerBox();
}
function updateVideoStatus(event){
  const video=$('mp-video'),note=$('mp-video-note');
  note.dataset.error=String(!!video.error);
  if(!video.dataset.src){note.textContent='No video is attached to this analysis. Open another half from Matches to review footage.';return;}
  if(video.error){
    $('mp-video-tools').open=true;
    const messages={1:'Video loading was interrupted.',2:'The connection to the footage failed.',3:'This browser could not decode the footage.',4:'The footage is unavailable or this browser cannot play its format.'};
    note.textContent=(messages[video.error.code]||'The footage could not load.')+' Try Reload footage or Open video.';return;
  }
  if(video.seeking){note.textContent='Loading the selected moment…';return;}
  if(event?.type==='waiting'||video.readyState<2){note.textContent='Loading footage… If it stays blank, try Reload footage or Open video.';return;}
  note.textContent=(video.paused?'Footage ready · press Play. ':'Playing footage. ')+'The yellow box follows the selected identity.';
}
function playPlayerVideo(){
  const video=$('mp-video'),src=video.dataset.src;
  return video.play().catch(error=>{
    if(video.dataset.src!==src||error.name==='AbortError')return;
    if(video.error)updateVideoStatus();
    else $('mp-video-note').textContent='Playback did not start. Press Play, or try Reload footage / Open video.';
  });
}
for(const event of ['loadstart','loadeddata','canplay','waiting','playing','pause','seeking','seeked','error'])$('mp-video').addEventListener(event,updateVideoStatus);
// Chrome can stall at an exact frame boundary (reproduced at 8.56s in both
// MKV and MP4). Retry a stalled seek once, 1ms later, within the same frame.
let seekRecoveryTimer=null,seekRecoveryAttempt=null;
function clearSeekRecovery(){clearTimeout(seekRecoveryTimer);seekRecoveryTimer=null;seekRecoveryAttempt=null;}
$('mp-video').addEventListener('seeking',()=>{
  const video=$('mp-video'),target=video.currentTime,source=video.dataset.src;
  clearTimeout(seekRecoveryTimer);
  if(seekRecoveryAttempt?.source===source&&Math.abs(seekRecoveryAttempt.target-target)<.02)return;
  seekRecoveryAttempt=null;
  seekRecoveryTimer=setTimeout(()=>{
    if(video.dataset.src!==source||video.error||!video.seeking||Math.abs(video.currentTime-target)>.0001)return;
    const retry=Math.min(Number.isFinite(video.duration)?video.duration-.001:Infinity,target+.001);
    if(retry<=target)return;
    seekRecoveryAttempt={source,target};video.currentTime=retry;
  },2500);
});
for(const event of ['seeked','loadstart','emptied','error'])$('mp-video').addEventListener(event,clearSeekRecovery);
$('mp-video').addEventListener('loadedmetadata',()=>{
  const video=$('mp-video');
  if(video.dataset.seek!=null){const target=Number(video.dataset.seek);delete video.dataset.seek;video.currentTime=target;}
});
$('mp-video-reload').onclick=()=>{
  const video=$('mp-video');if(!video.dataset.src)return;
  const target=Number(video.dataset.seek??video.currentTime);
  video.pause();video.dataset.seek=String(target);video.load();updateVideoStatus();
};
$('mp-watch').onclick=()=>{
  chooseEvidenceView('video');
  document.querySelector('.watch-panel').scrollIntoView({block:'start'});
  playPlayerVideo();
};
function drawPlayerBox(){
  const video=$('mp-video'),canvas=$('mp-video-overlay'),c=canvas.getContext('2d');
  c.clearRect(0,0,canvas.width,canvas.height);
  const times=matchState.boxTimes||[];if(!times.length)return;
  const t=video.currentTime;let lo=0,hi=times.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<t)lo=mid+1;else hi=mid;}
  const k=[lo-1,lo].filter(i=>i>=0).sort((a,b)=>Math.abs(times[a]-t)-Math.abs(times[b]-t))[0];
  if(k==null||Math.abs(times[k]-t)>Math.max(.1,.6/(state.manifest?.sampling_hz||12.5)))return;
  const [,x,y,w,h]=matchState.boxes[k];c.strokeStyle='#ffd23f';c.lineWidth=4;c.strokeRect(x-4,y-4,w+8,h+8);
}
function videoLoop(){drawPlayerBox();if(!$('mp-video').paused)requestAnimationFrame(videoLoop);}
$('mp-video').addEventListener('play',()=>requestAnimationFrame(videoLoop));
$('mp-video').addEventListener('seeked',drawPlayerBox);


$('team-names').onsubmit=handler(async e=>{e.preventDefault();await post(matchBase()+'/match/team-names',{A:$('team-a-name').value,B:$('team-b-name').value});notice('Team names saved.');await refreshSources(state.manifest.id);});
$('mp-merge').onsubmit=handler(async e=>{e.preventDefault();const r=await post(matchBase()+'/match/identities',{source:state.pid,target:$('mp-merge-target').value.trim()});await monitor(r.job_id,state.manifest.id);});
$('match-sort').onchange=()=>{matchState.sort=$('match-sort').value;renderMatchPlayers();};
$('match-basis').onchange=()=>{matchState.basis=$('match-basis').value;renderMatchPlayers();if(matchState.data&&state.pid){$('mp-stats').innerHTML=playerStats(matchState.data.players.find(x=>x.identity===state.pid)||{},true);if(eventMap.data)renderEventMap();}};
$('match-team-filter').onchange=()=>{matchState.team=$('match-team-filter').value;renderMatchPlayers();};
document.querySelectorAll('[data-goto]').forEach(a=>a.onclick=handler(async e=>{e.preventDefault();await changeTab(a.dataset.goto);}));

window.addEventListener('profilechange',handler(async()=>{if(state.tab!=='match')return;if(matchState.data?.id!==state.manifest?.id)await loadMatch();else await showMatchPlayer();}));
window.addEventListener('tabchange',handler(async()=>{if(state.tab==='match'){if(matchState.data?.id!==state.manifest?.id)await loadMatch();else await showMatchPlayer();}if(state.tab==='data')await loadMatchLibrary();}));
