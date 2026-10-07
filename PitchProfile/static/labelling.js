'use strict';

// Player evidence is loaded per dataset AND player; async responses never change their owner.
const reviewDesk={key:null,ready:false,loadFailed:false,spans:[],library:[]};
const draftKey=(dataset,pid)=>dataset+':'+pid;
function currentMatch(){return isMatch()&&matchState.data?.id===state.manifest?.id?matchState.data:null;}
function queuePlayers(){
  const data=currentMatch(),q=$('player-search').value.trim().toLowerCase(),team=$('queue-team').value;
  if(!data)return state.profiles.filter(p=>(p.name+' '+p.team+' '+p.player_id).toLowerCase().includes(q));
  return data.players.filter(p=>(!team||p.team===team)&&(!q||(p.name+' '+p.identity+' '+(data.teams?.[p.team]?.name||p.team)).toLowerCase().includes(q)))
    .sort((a,b)=>(b.visible_seconds||0)-(a.visible_seconds||0)||a.identity.localeCompare(b.identity));
}
function renderPlayers(){
  const data=currentMatch(),rows=queuePlayers();
  $('queue-filters').hidden=!data;
  if(data){
    const team=$('queue-team').value;
    $('queue-team').innerHTML='<option value="">Both teams</option>'+Object.entries(data.teams||{}).map(([id,t])=>`<option value="${esc(id)}">${esc(t.name||id)}</option>`).join('');
    $('queue-team').value=team in (data.teams||{})?team:'';
  }
  $('queue-count').textContent=rows.length+' player'+(rows.length===1?'':'s')+' in this queue';
  $('players').innerHTML=rows.length?rows.map(p=>{
    const pid=p.identity||p.player_id;
    return `<button type="button" class="player ${pid===state.pid?'active':''}" data-player="${esc(pid)}" aria-current="${pid===state.pid?'true':'false'}">${data?`<span class="queue-shirt">${p.jersey??'—'}</span>`:''}<strong>${esc(p.name)}</strong>${data?`<small>${esc(p.team_name||data.teams?.[p.team]?.name||p.team)}</small><span class="queue-meta"><small>${clock(p.visible_seconds)} visible</small></span>`:`<small>${esc(p.role||p.team)}</small>`}</button>`;
  }).join(''):'<p class="empty">No players match these filters. Try another team.</p>';
  $('players').querySelectorAll('button').forEach(b=>b.onclick=handler(()=>selectPlayer(b.dataset.player)));
  updateQueueNavigation();
}
function updateQueueNavigation(){
  const rows=queuePlayers(),i=rows.findIndex(p=>(p.identity||p.player_id)===state.pid);
  $('mp-prev').disabled=!rows.length||i===0;$('mp-next').disabled=!rows.length||(rows.length===1&&i===0);
}
async function advancePlayer(direction=1,ids=null){
  const rows=ids||queuePlayers().map(p=>p.identity||p.player_id),index=rows.indexOf(state.pid);
  const next=direction<0?rows[index>0?index-1:0]:rows[index+1]||(index<0?rows[0]:null);
  if(next&&next!==state.pid)await selectPlayer(next);
  else notice('You have reached the end of this queue. Change the filters or choose another match.');
}

function showEvidenceError(message){
  reviewDesk.loadFailed=true;
  $('mp-loading').hidden=false;$('mp-loading').textContent=message;
  $('mp-retry').hidden=false;
}
function setReviewLoading(loading){
  $('mp-retry').hidden=true;
  if(loading)$('mp-loading').textContent='Loading player evidence…';
  $('mp-loading').hidden=!loading;$('label-workbench').classList.toggle('loading',loading);
  $('label-workbench').inert=loading;$('label-workbench').setAttribute('aria-busy',String(loading));
}
window.addEventListener('profilechanging',()=>{
  reviewDesk.ready=false;reviewDesk.key=null;++matchState.playerRequest;++matchState.request;
  $('mp-video').pause();matchState.boxes=[];matchState.boxTimes=[];drawPlayerBox();setReviewLoading(true);
  $('mp-watch').disabled=true;
});

async function showMatchPlayer(){
  const data=currentMatch(),pid=state.pid,request=++matchState.playerRequest,p=data?.players.find(x=>x.identity===pid);
  // Reopening the tab keeps the loaded player and its scroll position.
  if(p&&reviewDesk.ready&&reviewDesk.key===draftKey(data.id,pid))return;
  $('match-player').hidden=!p;if(!p){reviewDesk.ready=false;return;}
  setReviewLoading(true);reviewDesk.ready=false;
  $('mp-team').textContent=(data.teams?.[p.team]?.name||p.team)+' · '+(p.role==='goalkeeper'?'Goalkeeper':'Outfield player');
  $('mp-name').textContent=p.name;
  $('mp-sub').textContent=`${clock(p.visible_seconds)} on screen · ${p.jersey!=null?'Shirt number read automatically':'Identity inferred from tactical position'} · verify in the footage`;
  renderMatchPlayers();renderPlayers();
  const root=matchBase(),key=draftKey(data.id,pid);
  try{
    const [crops,boxes]=await Promise.all([
      api(root+'/match/crops/'+encodeURIComponent(pid)),api(root+'/match/boxes/'+encodeURIComponent(pid))
    ]);
    if(request!==matchState.playerRequest||data.id!==state.manifest?.id||pid!==state.pid)return;
    reviewDesk.key=key;
    $('mp-stats').innerHTML=playerStats(p,true);pitch($('mp-heatmap'),state.player?.player_id===pid?state.player.heatmap:null);
    $('movement-coverage').textContent=(state.player?.direction_known?'Attacking right. ':'Attack direction unconfirmed. ')+"Whole-half movement, measured only while this player is visible. Event filters do not change this view.";
    const pos=p.positional||{};
    $('mp-zones').innerHTML=['defensive','middle','attacking'].map(k=>`<div>${k[0].toUpperCase()+k.slice(1)} third<strong>${pct(pos[k+'_third'])}</strong></div>`).join('');
    $('mp-crops').innerHTML=crops.crops.length?crops.crops.map(u=>`<img src="${esc(u)}" alt="Automatic thumbnail of ${esc(p.name)}" loading="lazy">`).join(''):'<p class="empty">No readable thumbnails for this identity. Check the highlighted player in the video.</p>';
    loadPlayerVideo(boxes);buildAppearanceTimeline();
    reviewDesk.ready=true;setReviewLoading(false);
    if(reviewDesk.loadFailed){clearNotice();reviewDesk.loadFailed=false;}
    if(boxes.boxes.length)seekVideo(boxes.boxes[0][0],false,0);
    window.dispatchEvent(new Event('reviewready'));
  }catch(error){
    if(request!==matchState.playerRequest)return;
    showEvidenceError('Could not load player evidence.');
    notice(error.message,true);
  }
}

function videoBounds(){const start=Number(state.manifest?.source_offset_s||0);return {start,end:start+Number(state.manifest?.duration_seconds||0)};}
function seekVideo(time,play=false,lead=0){
  const video=$('mp-video');if(!video.dataset.src)return;
  const {start,end}=videoBounds(),target=Math.max(start,Math.min(end-.05,Number(time)-lead));
  if(video.readyState>=1){delete video.dataset.seek;video.currentTime=target;}
  else video.dataset.seek=String(target);
  updatePlayback();if(play)playPlayerVideo();
}
function buildAppearanceTimeline(){
  const spans=[],step=1/(state.manifest?.sampling_hz||12.5);
  for(const t of matchState.boxTimes||[]){const last=spans.at(-1);if(last&&t-last.end<=.75)last.end=t+step;else spans.push({start:t,end:t+step});}
  reviewDesk.spans=spans;const {start,end}=videoBounds(),duration=Math.max(1,end-start);
  $('mp-timeline').innerHTML=spans.map((s,i)=>`<button type="button" data-span="${i}" title="${clock(s.start)}–${clock(s.end)}" aria-label="Watch appearance at ${clock(s.start)}" style="left:${Math.max(0,(s.start-start)/duration*100)}%;width:${Math.max(.15,(s.end-s.start)/duration*100)}%"></button>`).join('')+'<span class="playhead"></span>';
  $('mp-visibility').textContent=spans.length?`${spans.length} appearances · click a green section to watch`:'No image boxes available for this player. Use events to find footage.';
  $('mp-appearance').disabled=!spans.length;
  for(const id of ['mp-back','mp-forward','mp-speed'])$(id).disabled=!$('mp-video').dataset.src;
  updatePlayback();
}
function updatePlayback(){
  const video=$('mp-video'),{start,end}=videoBounds();$('mp-playback-time').textContent=clock(video.currentTime)+' / '+clock(end);
  const cursor=$('mp-timeline').querySelector('.playhead');if(cursor)cursor.style.left=Math.max(0,Math.min(100,(video.currentTime-start)/Math.max(1,end-start)*100))+'%';
}
function toggleVideo(){const v=$('mp-video');if(v.paused)playPlayerVideo();else v.pause();}

$('queue-team').onchange=renderPlayers;
$('mp-prev').onclick=handler(()=>advancePlayer(-1));$('mp-next').onclick=handler(()=>advancePlayer());
$('mp-back').onclick=()=>seekVideo($('mp-video').currentTime-5);$('mp-forward').onclick=()=>seekVideo($('mp-video').currentTime+5);
$('mp-speed').onchange=()=>$('mp-video').playbackRate=Number($('mp-speed').value);
$('mp-appearance').onclick=()=>{const s=reviewDesk.spans.find(s=>s.start>$('mp-video').currentTime+1)||reviewDesk.spans[0];if(s)seekVideo(s.start,true);};
$('mp-timeline').onclick=e=>{const b=e.target.closest('[data-span]');if(b)seekVideo(reviewDesk.spans[Number(b.dataset.span)].start,true);};
for(const event of ['timeupdate','play','pause','loadedmetadata'])$('mp-video').addEventListener(event,updatePlayback);
window.addEventListener('keydown',e=>{
  if(state.tab!=='match'||!reviewDesk.ready||document.querySelector('dialog[open]'))return;
  if(e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,textarea,select,[contenteditable=true]'))return;
  if(e.key===' '&&e.target.closest('button,summary,video'))return;
  if([' ','k','j','l'].includes(e.key.toLowerCase()))e.preventDefault();
  if(e.key===' '||e.key.toLowerCase()==='k')toggleVideo();
  if(e.key.toLowerCase()==='j')seekVideo($('mp-video').currentTime-5);
  if(e.key.toLowerCase()==='l')seekVideo($('mp-video').currentTime+5);
});

async function loadMatchLibrary(){
  const r=await api('/api/matches/library');reviewDesk.library=r.halves;renderMatchLibrary();
}
function renderMatchLibrary(){
  const q=$('match-search').value.toLowerCase(),rows=reviewDesk.library.filter(h=>h.title.toLowerCase().includes(q)).sort((a,b)=>Number(b.analysed)-Number(a.analysed)||a.title.localeCompare(b.title));
  $('library-count').textContent=rows.length+' halves · '+reviewDesk.library.filter(h=>h.analysed).length+' ready to review';
  $('learn-library').innerHTML=rows.length?'<table><thead><tr><th>Match / half</th><th>Status</th><th>Action</th></tr></thead><tbody>'+rows.map(h=>`<tr><td>${esc(h.title)}</td><td><span class="badge">${h.analysed?'Ready to review':h.detected?'Finish analysis':'Not analysed'}</span></td><td>${h.analysed?`<button type="button" data-open="${esc(h.dataset_id)}" class="primary">Review players →</button>`:`<button type="button" data-analyse="${esc(h.library_id)}">${h.detected?'Finish analysis':'Analyse on GPU'}</button>`}</td></tr>`).join('')+'</tbody></table>':'<p class="empty">No matching downloaded halves. Clear the search or check your data drive.</p>';
  $('learn-library').querySelectorAll('[data-open]').forEach(b=>b.onclick=handler(async()=>{await selectDataset(b.dataset.open);await changeTab('match');}));
  $('learn-library').querySelectorAll('[data-analyse]').forEach(b=>b.onclick=handler(async()=>{b.disabled=true;try{const r=await post('/api/matches/analyse',{library_id:b.dataset.analyse});notice('Match analysis started on your GPU. Progress appears above.');await monitor(r.job_id,r.dataset_id);await loadMatchLibrary();}finally{b.disabled=false;}}));
}
$('match-search').oninput=renderMatchLibrary;$('library-refresh').onclick=handler(loadMatchLibrary);
document.addEventListener('click',e=>{const menu=document.querySelector('.tools-menu');if(menu.open&&!menu.contains(e.target))menu.open=false;});

$('mp-retry').onclick=handler(()=>selectDataset($('dataset-select').value,state.pid));
