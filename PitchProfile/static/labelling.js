'use strict';

// A draft belongs to a dataset AND a player; async responses never change its owner.
const reviewDesk={key:null,ready:false,dirty:false,version:0,busy:false,evidence:[],groups:{},saved:null,events:[],spans:[],library:[]};
const draftKey=(dataset,pid)=>'pitchprofile-draft-v1:'+dataset+':'+pid;
function readDraft(key){try{const d=JSON.parse(safeLocalGet(key));return d?.payload&&typeof d.payload.labels==='object'?d:null;}catch{return null;}}
function removeDraft(key){try{localStorage.removeItem(key);}catch{/* The saved server label remains authoritative. */}}
function currentMatch(){return isMatch()&&matchState.data?.id===state.manifest?.id?matchState.data:null;}
function queuePlayers(){
  const data=currentMatch(),q=$('player-search').value.trim().toLowerCase(),team=$('queue-team').value,status=$('queue-status').value;
  if(!data)return state.profiles.filter(p=>(p.name+' '+p.team+' '+p.player_id).toLowerCase().includes(q));
  return data.players.filter(p=>{
    const draft=readDraft(draftKey(data.id,p.identity));
    return (!team||p.team===team)&&(!q||(p.name+' '+p.identity+' '+(data.teams?.[p.team]?.name||p.team)).toLowerCase().includes(q))&&
      (status==='all'||status==='unlabelled'&&!p.label||status==='labelled'&&p.label||status==='draft'&&draft);
  }).sort((a,b)=>(b.visible_seconds||0)-(a.visible_seconds||0)||a.identity.localeCompare(b.identity));
}
function renderPlayers(){
  const data=currentMatch(),rows=queuePlayers();
  $('queue-filters').hidden=!data;$('queue-progress').hidden=!data;
  if(data){
    const team=$('queue-team').value;
    $('queue-team').innerHTML='<option value="">Both teams</option>'+Object.entries(data.teams||{}).map(([id,t])=>`<option value="${esc(id)}">${esc(t.name||id)}</option>`).join('');
    $('queue-team').value=team in (data.teams||{})?team:'';
    const n=data.players.filter(p=>p.label).length;
    $('queue-progress').innerHTML=`<strong>${n} / ${data.players.length}</strong> players rated<progress max="${data.players.length||1}" value="${n}" aria-label="Players rated"></progress>`;
  }
  $('queue-count').textContent=rows.length+' player'+(rows.length===1?'':'s')+' in this queue';
  $('players').innerHTML=rows.length?rows.map(p=>{
    const pid=p.identity||p.player_id,draft=data&&readDraft(draftKey(data.id,pid)),status=draft?'Draft':p.label?'Rated':'Not rated yet';
    return `<button type="button" class="player ${pid===state.pid?'active':''}" data-player="${esc(pid)}" aria-current="${pid===state.pid?'true':'false'}">${data?`<span class="queue-shirt">${p.jersey??'—'}</span>`:''}<strong>${esc(p.name)}</strong>${data?`<span class="queue-dot" aria-hidden="true">${draft?'◌':p.label?'✓':''}</span><small>${esc(p.team_name||data.teams?.[p.team]?.name||p.team)}</small><span class="queue-meta"><small>${clock(p.visible_seconds)} visible</small><span class="queue-tag ${draft?'draft':p.label?'labelled':''}">${status}</span></span>`:`<small>${esc(p.role||p.team)}</small>`}</button>`;
  }).join(''):'<p class="empty">No players match these filters. Try another team or review status.</p>';
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

function reviewPayload(){return {labeler:$('mp-labeler').value,position_group:$('mp-group').value,labels:labelValues(),notes:$('mp-notes').value,evidence:reviewDesk.evidence.map(e=>({...e}))};}
function persistReview(){
  if(!reviewDesk.ready||!reviewDesk.dirty)return;
  const payload=reviewPayload();reviewDesk.groups[payload.position_group]=payload.labels;
  const stored=safeLocalSet(reviewDesk.key,JSON.stringify({payload,groups:reviewDesk.groups,baseUpdated:reviewDesk.saved?.updated||null}));
  $('mp-draft-status').textContent=stored?'Draft saved on this browser · save to add it to your labels':'Browser storage unavailable · save before leaving this player';
  $('mp-draft-status').dataset.dirty='true';
  $('mp-save-state').textContent=stored?'Local draft · not yet saved to the dataset':'Unsaved changes · browser storage unavailable';
}
function markReviewDirty(){
  if(!reviewDesk.ready)return;
  const wasDirty=reviewDesk.dirty;reviewDesk.dirty=true;reviewDesk.version++;persistReview();
  $('mp-discard').disabled=false;
  if(!wasDirty)renderPlayers();
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
  persistReview();reviewDesk.ready=false;reviewDesk.key=null;++matchState.playerRequest;++matchState.request;
  $('mp-video').pause();matchState.boxes=[];matchState.boxTimes=[];drawPlayerBox();setReviewLoading(true);
  $('mp-watch').disabled=true;
});

async function showMatchPlayer(){
  const data=currentMatch(),pid=state.pid,request=++matchState.playerRequest,p=data?.players.find(x=>x.identity===pid);
  // Reopening the tab should retain an in-progress form and its scroll position.
  if(p&&reviewDesk.ready&&reviewDesk.key===draftKey(data.id,pid))return;
  $('match-player').hidden=!p;if(!p){reviewDesk.ready=false;return;}
  setReviewLoading(true);reviewDesk.ready=false;
  $('mp-team').textContent=(data.teams?.[p.team]?.name||p.team)+' · '+(p.role==='goalkeeper'?'Goalkeeper':'Outfield player');
  $('mp-name').textContent=p.name;
  $('mp-sub').textContent=`${clock(p.visible_seconds)} on screen · ${p.jersey!=null?'Shirt number read automatically':'Identity inferred from tactical position'} · verify in the footage`;
  $('mp-label-state').textContent=p.label?'Rated by '+p.label.labeler:'Not rated yet';
  renderMatchPlayers();renderPlayers();
  const root=matchBase(),key=draftKey(data.id,pid);
  try{
    const [crops,archetype,boxes]=await Promise.all([
      api(root+'/match/crops/'+encodeURIComponent(pid)),
      api(root+'/players/'+encodeURIComponent(pid)+'/archetype'),api(root+'/match/boxes/'+encodeURIComponent(pid)),catalogueReady
    ]);
    if(request!==matchState.playerRequest||data.id!==state.manifest?.id||pid!==state.pid)return;
    reviewDesk.key=key;reviewDesk.saved=archetype.label;reviewDesk.version=0;reviewDesk.busy=false;
    reviewDesk.groups={};
    const draft=readDraft(key),values=draft?.payload||archetype.label||{};
    const group=values.position_group||p.position_group||'central_midfield';
    $('mp-group').innerHTML=intervalState.catalogue.position_groups.map(g=>`<option value="${esc(g.id)}">${esc(g.name)}</option>`).join('');
    $('mp-group').value=group;if(!$('mp-group').value)$('mp-group').value='central_midfield';
    reviewDesk.group=$('mp-group').value;reviewDesk.groups=draft?.groups||{};
    $('mp-labeler').value=values.labeler||safeLocalGet('pitchprofile-labeller')||'';
    $('mp-notes').value=values.notes||'';
    reviewDesk.evidence=(values.evidence||[]).map(e=>({...e}));
    renderRoleSliders(reviewDesk.group,values.labels||{});renderBookmarks();
    $('mp-roles').parentElement.scrollTop=0;
    $('mp-delete-confirm').hidden=true;$('mp-unlabel').disabled=!archetype.label;
    $('mp-save').disabled=false;$('mp-save-next').disabled=false;
    $('mp-estimate-panel').open=false;renderArchetypePrediction(archetype.prediction);
    $('mp-stats').innerHTML=playerStats(p,true);pitch($('mp-heatmap'),state.player?.player_id===pid?state.player.heatmap:null);
    $('movement-coverage').textContent=(state.player?.direction_known?'Attacking right. ':'Attack direction unconfirmed. ')+"Whole-half movement, measured only while this player is visible. Event filters do not change this view.";
    const pos=p.positional||{};
    $('mp-zones').innerHTML=['defensive','middle','attacking'].map(k=>`<div>${k[0].toUpperCase()+k.slice(1)} third<strong>${pct(pos[k+'_third'])}</strong></div>`).join('');
    $('mp-crops').innerHTML=crops.crops.length?crops.crops.map(u=>`<img src="${esc(u)}" alt="Automatic thumbnail of ${esc(p.name)}" loading="lazy">`).join(''):'<p class="empty">No readable thumbnails for this identity. Check the highlighted player in the video.</p>';
    loadPlayerVideo(boxes);buildAppearanceTimeline();
    reviewDesk.ready=true;reviewDesk.dirty=!!draft;setReviewLoading(false);
    if(reviewDesk.loadFailed){clearNotice();reviewDesk.loadFailed=false;}
    $('mp-draft-status').dataset.dirty=String(!!draft);
    $('mp-draft-status').textContent=draft?(draft.baseUpdated!==(archetype.label?.updated||null)?'Restored draft · the saved label has changed; check before replacing it':'Restored your local draft'):archetype.label?'Saved assessment · changes create a new draft':'No label yet · your edits are kept as a local draft';
    $('mp-save-state').textContent=draft?'Local draft · save when ready':archetype.label?'Saved to the dataset':'Leave unknown when unsure';
    $('mp-discard').disabled=!draft;
    if(boxes.boxes.length)seekVideo(boxes.boxes[0][0],false,0);
    window.dispatchEvent(new Event('reviewready'));
  }catch(error){
    if(request!==matchState.playerRequest)return;
    showEvidenceError('Could not load player evidence. Your saved ratings and draft are retained.');
    notice(error.message,true);
  }
}

function renderRoleSliders(group,values){
  $('mp-roles').innerHTML=compatibleRoles(group).map((r,index)=>{
    const value=values[r.id],known=value!=null;
    return `<div class="role-card" data-role="${esc(r.id)}" data-known="${known}"><details class="role-entry" ${index===0?'open':''}><summary class="role-title"><span class="role-name">${esc(r.name)}</span><output for="rating-${esc(r.id)}">${known?value+'%':'Unknown'}</output></summary><div class="role-body"><p>${esc(r.definition)}</p>${roleExamples(r)}<p><strong>Look for:</strong> ${esc(r.evidence_cues)}</p>${r.trainable?'':'<p>Descriptive role; excluded from model training.</p>'}<label class="small" for="rating-${esc(r.id)}">How strongly was this behaviour observed?</label><input id="rating-${esc(r.id)}" type="range" min="0" max="100" step="1" value="${known?value:0}" aria-valuetext="${known?value+' percent':'Unknown; move to rate'}"><div class="rating-presets">${[[0,'0 · None'],[50,'50 · Some'],[100,'100 · Strong'],['unknown','Unknown']].map(([v,title])=>`<button type="button" data-value="${v}" aria-pressed="${v==='unknown'?!known:known&&Number(value)===v}" aria-label="${esc(r.name)}: ${title}">${title}</button>`).join('')}</div></div></details></div>`;
  }).join('');
  $('mp-roles').querySelectorAll('.role-card').forEach(row=>{
    row.querySelector('details').addEventListener('toggle',()=>{
      if(row.querySelector('details').open)$('mp-roles').querySelectorAll('.role-card details').forEach(d=>{if(d!==row.querySelector('details'))d.open=false;});
    });
    row.querySelector('input').oninput=e=>setRoleRating(row,Number(e.target.value));
    row.querySelectorAll('button').forEach(b=>b.onclick=()=>setRoleRating(row,b.dataset.value==='unknown'?null:Number(b.dataset.value)));
  });
  updateLabelMixture();
}
function setRoleRating(row,value){
  row.dataset.known=String(value!=null);const slider=row.querySelector('input');
  if(value!=null)slider.value=value;
  slider.setAttribute('aria-valuetext',value==null?'Unknown; move to rate':value+' percent');
  row.querySelector('output').textContent=value==null?'Unknown':value+'%';
  row.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(value==null?b.dataset.value==='unknown':Number(b.dataset.value)===value)));
  updateLabelMixture();markReviewDirty();
}
function labelValues(){return Object.fromEntries([...$('mp-roles').querySelectorAll('.role-card')].map(row=>[row.dataset.role,row.dataset.known==='true'?Number(row.querySelector('input').value):null]));}
function updateLabelMixture(){
  const all=Object.entries(labelValues()),values=all.filter(([,v])=>v>0),total=values.reduce((s,[,v])=>s+v,0)||1;
  $('mp-role-count').textContent=`${all.filter(([,v])=>v!=null).length} of ${all.length} compatible roles rated · 42 roles across all groups`;
  $('mp-mixture').innerHTML=values.length?bars(values.sort((a,b)=>b[1]-a[1]).map(([k,v])=>({label:names[k]||k,value:v/total*100}))):'<p class="small muted">Rate an observed role above to see its share of your ratings.</p>';
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
  for(const id of ['mp-back','mp-forward','mp-speed','mp-bookmark'])$(id).disabled=!$('mp-video').dataset.src;
  updatePlayback();
}
function updatePlayback(){
  const video=$('mp-video'),{start,end}=videoBounds();$('mp-playback-time').textContent=clock(video.currentTime)+' / '+clock(end);
  const cursor=$('mp-timeline').querySelector('.playhead');if(cursor)cursor.style.left=Math.max(0,Math.min(100,(video.currentTime-start)/Math.max(1,end-start)*100))+'%';
}
function toggleVideo(){const v=$('mp-video');if(v.paused)playPlayerVideo();else v.pause();}
function renderBookmarks(){
  $('mp-bookmarks').innerHTML=reviewDesk.evidence.length?reviewDesk.evidence.map((e,i)=>`<div class="bookmark-row"><button type="button" data-evidence-seek="${i}" aria-label="Watch evidence at ${clock(e.time_s)}">▶ ${clock(e.time_s)}</button><textarea rows="1" maxlength="500" data-evidence-note="${i}" aria-label="Evidence note at ${clock(e.time_s)}" placeholder="What did the player do?">${esc(e.note)}</textarea><button type="button" class="bookmark-remove" data-evidence-remove="${i}" aria-label="Remove evidence at ${clock(e.time_s)}">×</button></div>`).join(''):'<p class="bookmark-empty">No bookmarks yet. Pause at a useful moment and press B.</p>';
}
function addBookmark(){
  if(!reviewDesk.ready||!$('mp-video').dataset.src)return;
  if(reviewDesk.evidence.length>=60){notice('This label already has 60 bookmarks. Remove one before adding another.',true);return;}
  const {start,end}=videoBounds(),time_s=Math.min(end,Math.max(start,Math.round($('mp-video').currentTime*10)/10));
  $('mp-video').pause();reviewDesk.evidence.push({time_s,note:''});renderBookmarks();markReviewDirty();
  $('mp-bookmarks').querySelectorAll('textarea')[reviewDesk.evidence.length-1].focus({preventScroll:true});
}

async function savePlayerLabel(next){
  if(!reviewDesk.ready||reviewDesk.busy)return;
  if(!$('mp-label').reportValidity())return;
  updateIdentityGroupWarning();if(!$('mp-group-warning').hidden){$('mp-group').focus();return;}
  const payload=reviewPayload();
  if(!Object.values(payload.labels).some(v=>v!=null)){notice('Rate at least one observed role. You can leave other roles unknown, or use Next player to skip.',true);$('mp-roles').querySelector('input')?.focus();return;}
  const key=reviewDesk.key,dataset=state.manifest.id,pid=state.pid,version=reviewDesk.version;
  const queue=queuePlayers().map(p=>p.identity||p.player_id);
  reviewDesk.busy=true;$('mp-save').disabled=true;$('mp-save-next').disabled=true;$('mp-save-state').textContent='Saving…';
  try{
    const saved=await post(matchBase()+'/players/'+encodeURIComponent(pid)+'/archetype',payload);
    safeLocalSet('pitchprofile-labeller',payload.labeler);
    const draft=readDraft(key);if(draft&&JSON.stringify(draft.payload)===JSON.stringify(payload))removeDraft(key);
    if(matchState.data?.id===dataset){const p=matchState.data.players.find(p=>p.identity===pid);if(p)p.label=saved;renderMatchPlayers();}
    if(reviewDesk.key!==key){renderPlayers();return;}
    reviewDesk.saved=saved;$('mp-unlabel').disabled=false;$('mp-label-state').textContent='Rated by '+saved.labeler;
    if(reviewDesk.version===version){
      removeDraft(key);reviewDesk.dirty=false;$('mp-draft-status').dataset.dirty='false';$('mp-draft-status').textContent='Saved assessment';$('mp-save-state').textContent='Saved to the dataset';$('mp-discard').disabled=true;
      notice('Label and '+payload.evidence.length+' evidence bookmark'+(payload.evidence.length===1?'':'s')+' saved.');
      renderPlayers();if(next)await advancePlayer(1,queue);
    }else{persistReview();notice('Label saved. Your newer edits are still a local draft.');}
  }catch(error){
    if(reviewDesk.key===key){reviewDesk.dirty=true;persistReview();$('mp-save-state').textContent='Save failed · your draft is kept here';}
    throw error;
  }finally{if(reviewDesk.key===key){reviewDesk.busy=false;$('mp-save').disabled=false;$('mp-save-next').disabled=false;}}
}

$('queue-team').onchange=renderPlayers;$('queue-status').onchange=renderPlayers;
$('mp-prev').onclick=handler(()=>advancePlayer(-1));$('mp-next').onclick=handler(()=>advancePlayer());
$('mp-group').onchange=()=>{reviewDesk.groups[reviewDesk.group]=labelValues();reviewDesk.group=$('mp-group').value;renderRoleSliders(reviewDesk.group,reviewDesk.groups[reviewDesk.group]||{});markReviewDirty();};
$('mp-labeler').oninput=()=>{safeLocalSet('pitchprofile-labeller',$('mp-labeler').value);markReviewDirty();};
$('mp-notes').oninput=markReviewDirty;
$('mp-label').onsubmit=handler(async e=>{e.preventDefault();await savePlayerLabel(e.submitter?.id==='mp-save-next');});
$('mp-discard').onclick=handler(async()=>{removeDraft(reviewDesk.key);reviewDesk.ready=false;reviewDesk.dirty=false;await showMatchPlayer();renderPlayers();});
$('mp-unlabel').onclick=()=>$('mp-delete-confirm').hidden=false;
$('mp-delete-no').onclick=()=>$('mp-delete-confirm').hidden=true;
$('mp-delete-yes').onclick=handler(async()=>{
  if(!reviewDesk.ready||reviewDesk.busy)return;
  const key=reviewDesk.key,root=matchBase(),pid=state.pid,id=state.manifest.id;
  reviewDesk.busy=true;$('mp-delete-yes').disabled=true;
  try{
    await api(root+'/players/'+encodeURIComponent(pid)+'/archetype',{method:'DELETE'});
    if(matchState.data?.id===id){const p=matchState.data.players.find(p=>p.identity===pid);if(p)p.label=null;}
    if(reviewDesk.key===key){reviewDesk.saved=null;$('mp-unlabel').disabled=true;$('mp-delete-confirm').hidden=true;$('mp-label-state').textContent='Not rated yet';markReviewDirty();}
    notice('Saved label removed. The current form is kept as a draft.');renderPlayers();
  }finally{if(reviewDesk.key===key)reviewDesk.busy=false;$('mp-delete-yes').disabled=false;}
});
$('mp-back').onclick=()=>seekVideo($('mp-video').currentTime-5);$('mp-forward').onclick=()=>seekVideo($('mp-video').currentTime+5);
$('mp-speed').onchange=()=>$('mp-video').playbackRate=Number($('mp-speed').value);
$('mp-appearance').onclick=()=>{const s=reviewDesk.spans.find(s=>s.start>$('mp-video').currentTime+1)||reviewDesk.spans[0];if(s)seekVideo(s.start,true);};
$('mp-timeline').onclick=e=>{const b=e.target.closest('[data-span]');if(b)seekVideo(reviewDesk.spans[Number(b.dataset.span)].start,true);};
$('mp-bookmark').onclick=addBookmark;
$('mp-bookmarks').oninput=e=>{if(e.target.matches('[data-evidence-note]')){reviewDesk.evidence[Number(e.target.dataset.evidenceNote)].note=e.target.value;markReviewDirty();}};
$('mp-bookmarks').onclick=e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.evidenceSeek!=null)seekVideo(reviewDesk.evidence[Number(b.dataset.evidenceSeek)].time_s,true,2);if(b.dataset.evidenceRemove!=null){reviewDesk.evidence.splice(Number(b.dataset.evidenceRemove),1);renderBookmarks();markReviewDirty();}};
for(const event of ['timeupdate','play','pause','loadedmetadata'])$('mp-video').addEventListener(event,updatePlayback);
window.addEventListener('beforeunload',persistReview);
window.addEventListener('keydown',e=>{
  if(state.tab!=='match'||!reviewDesk.ready||document.querySelector('dialog[open]'))return;
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();handler(()=>savePlayerLabel(true))();return;}
  if(e.ctrlKey||e.metaKey||e.altKey||e.target.closest('input,textarea,select,[contenteditable=true]'))return;
  if(e.key===' '&&e.target.closest('button,summary,video'))return;
  if(e.repeat&&e.key.toLowerCase()==='b')return;
  if([' ','k','j','l','b'].includes(e.key.toLowerCase()))e.preventDefault();
  if(e.key===' '||e.key.toLowerCase()==='k')toggleVideo();
  if(e.key.toLowerCase()==='j')seekVideo($('mp-video').currentTime-5);
  if(e.key.toLowerCase()==='l')seekVideo($('mp-video').currentTime+5);
  if(e.key.toLowerCase()==='b')addBookmark();
});

async function loadMatchLibrary(){
  const r=await api('/api/matches/library');reviewDesk.library=r.halves;renderMatchLibrary();
}
function renderMatchLibrary(){
  const q=$('match-search').value.toLowerCase(),rows=reviewDesk.library.filter(h=>h.title.toLowerCase().includes(q)).sort((a,b)=>Number(b.analysed)-Number(a.analysed)||a.title.localeCompare(b.title));
  $('library-count').textContent=rows.length+' halves · '+reviewDesk.library.filter(h=>h.analysed).length+' ready to label';
  $('learn-library').innerHTML=rows.length?'<table><thead><tr><th>Match / half</th><th>Status</th><th>Action</th></tr></thead><tbody>'+rows.map(h=>`<tr><td>${esc(h.title)}</td><td><span class="badge">${h.analysed?'Ready to label':h.detected?'Finish analysis':'Not analysed'}</span></td><td>${h.analysed?`<button type="button" data-open="${esc(h.dataset_id)}" class="primary">Label players →</button>`:`<button type="button" data-analyse="${esc(h.library_id)}">${h.detected?'Finish analysis':'Analyse on GPU'}</button>`}</td></tr>`).join('')+'</tbody></table>':'<p class="empty">No matching downloaded halves. Clear the search or check your data drive.</p>';
  $('learn-library').querySelectorAll('[data-open]').forEach(b=>b.onclick=handler(async()=>{await selectDataset(b.dataset.open);await changeTab('match');}));
  $('learn-library').querySelectorAll('[data-analyse]').forEach(b=>b.onclick=handler(async()=>{b.disabled=true;try{const r=await post('/api/matches/analyse',{library_id:b.dataset.analyse});notice('Match analysis started on your GPU. Progress appears above.');await monitor(r.job_id,r.dataset_id);await loadMatchLibrary();}finally{b.disabled=false;}}));
}
$('match-search').oninput=renderMatchLibrary;$('library-refresh').onclick=handler(loadMatchLibrary);
document.addEventListener('click',e=>{const menu=document.querySelector('.tools-menu');if(menu.open&&!menu.contains(e.target))menu.open=false;});

// A single page scroll keeps definitions and rating controls reachable on short screens.
$('mp-retry').onclick=handler(()=>selectDataset($('dataset-select').value,state.pid));
