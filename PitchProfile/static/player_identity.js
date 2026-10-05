'use strict';

const identityPicker={key:null,team:null,athlete:null,mode:'roster',busy:false,generation:0};
const identityPlayer=()=>currentMatch()?.players.find(p=>p.identity===state.pid);
const identityTeams=()=>currentMatch()?.match_context?.context?.teams||[];
const identityTeam=()=>identityTeams().find(t=>t.id===identityPicker.team);
const identityCandidate=()=>identityTeam()?.players.find(p=>p.id===identityPicker.athlete&&p.played);
const normalIdentityText=s=>String(s||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().trim();

function setIdentityRole(role){
  $('correction-role').value=role;
  document.querySelectorAll('[data-identity-role]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.identityRole===role)));
  updateIdentitySelection();
}
function setIdentityMode(mode){
  identityPicker.mode=mode;$('identify-roster').hidden=mode!=='roster';$('identify-quick').hidden=mode!=='quick';
  $('correction-shirt').disabled=mode!=='quick';$('correction-name').disabled=mode!=='quick';$('correction-note').disabled=mode!=='quick';
  updateIdentitySelection();
}
function updateIdentitySelection(){
  const candidate=identityCandidate(),quick=identityPicker.mode==='quick',number=$('correction-shirt').value;
  const other=quick||!candidate?[]:(currentMatch()?.players||[]).filter(p=>p.identity!==state.pid&&p.team_id===identityPicker.team&&p.lineup_player?.id===candidate.id);
  $('identify-duplicate').hidden=!other.length;$('identify-duplicate').textContent=other.length?`This person also appears in ${other.length} other player ${other.length===1?'entry':'entries'}. Their statistics will remain separate.`:'';
  $('identify-selection').textContent=quick?`${$('correction-role').value==='goalkeeper'?'Goalkeeper':'Outfield player'}${number?' · #'+number:''}${$('correction-name').value.trim()?' · '+$('correction-name').value.trim():''}`:
    candidate?`${identityTeam().name} · #${candidate.shirt??'?'} · ${candidate.name}`:identityTeam()?'Choose the player you see in the footage.':'Choose a team and player above.';
  $('identify-confirm').textContent=quick?'Save correction':candidate?'Confirm '+candidate.name:'Confirm player';
  $('identify-confirm').disabled=identityPicker.busy||(!quick&&!candidate);
}
function renderIdentityCandidates(){
  const team=identityTeam(),q=normalIdentityText($('identify-search').value).replace(/^#/,'');
  const people=(team?.players||[]).filter(p=>p.played&&(!q||(/^\d+$/.test(q)?p.shirt===Number(q):normalIdentityText(p.name).includes(q))))
    .sort((a,b)=>(a.shirt??100)-(b.shirt??100));
  $('identify-search').disabled=!team;
  $('identify-candidates').innerHTML=people.map(p=>`<button type="button" class="identify-candidate" data-athlete="${esc(p.id)}" aria-pressed="${p.id===identityPicker.athlete}" aria-label="Number ${p.shirt??'unknown'}, ${esc(p.name)}"><span class="candidate-portrait"><span>${p.shirt??'?'}</span>${p.headshot?`<img src="${esc(p.headshot)}" alt="" loading="lazy" referrerpolicy="no-referrer">`:''}</span><span class="candidate-info"><strong>#${p.shirt??'?'} · ${esc(p.name)}</strong><small>${esc(p.position||'Player')} · ${p.starter?'Starter':'Came on'}</small></span><span class="candidate-check" aria-hidden="true">${p.id===identityPicker.athlete?'✓':''}</span></button>`).join('');
  $('identify-list-note').textContent=!team?'Choose a team to see its players.':!people.length?'No match. Try another number or name, or enter what you know below.':`${people.length} ${people.length===1?'player':'players'} · only players who took part in this match`;
}
function renderIdentityTeams(){
  const teams=identityTeams();
  $('identify-no-roster').hidden=!!teams.length;
  $('identify-teams').innerHTML=teams.map(t=>`<button type="button" data-identity-team="${esc(t.id)}" aria-pressed="${identityPicker.team===t.id}">${teamCrest(t)}<span>${esc(t.name)}</span></button>`).join('');
  renderIdentityCandidates();updateIdentitySelection();
}
function showIdentityFrame(){
  const video=$('mp-video'),canvas=$('identify-frame'),ctx=canvas.getContext('2d'),t=video.currentTime;
  canvas.hidden=true;
  if(video.readyState>=2&&!video.seeking&&video.videoWidth){
    const box=(matchState.boxes||[]).reduce((best,b)=>!best||Math.abs(b[0]-t)<Math.abs(best[0]-t)?b:best,null);
    if(box&&Math.abs(box[0]-t)<=Math.max(.1,.6/(state.manifest?.sampling_hz||12.5))){
      const sx=video.videoWidth/($('mp-video-overlay').width||video.videoWidth),sy=video.videoHeight/($('mp-video-overlay').height||video.videoHeight);
      const x=Math.max(0,(box[1]-box[3]*.5)*sx),y=Math.max(0,(box[2]-box[4]*.15)*sy);
      const w=Math.min(video.videoWidth-x,box[3]*2*sx),h=Math.min(video.videoHeight-y,box[4]*1.3*sy);
      canvas.height=300;canvas.width=Math.max(1,Math.round(300*w/h));ctx.drawImage(video,x,y,w,h,0,0,canvas.width,canvas.height);
      $('identify-frame-note').textContent='Highlighted player · '+clock(t)+' · enlarged from paused footage';
    }else{
      canvas.width=480;canvas.height=Math.round(480*video.videoHeight/video.videoWidth);ctx.drawImage(video,0,0,canvas.width,canvas.height);
      $('identify-frame-note').textContent='Paused footage · '+clock(t)+' · no player box at this moment';
    }
    canvas.hidden=false;
  }else $('identify-frame-note').textContent='Example images below. Return to the footage to check the highlighted player.';
}
function openPlayerCorrection(notKeeper=false){
  if(!reviewDesk.ready||identityPicker.busy)return;
  const p=identityPlayer();if(!p)return;
  $('mp-video').pause();
  identityPicker.generation++;identityPicker.key=reviewDesk.key;identityPicker.team=p.team_id||null;identityPicker.athlete=null;
  $('identify-search').value='';$('correction-shirt').value=notKeeper?'':p.jersey??'';
  $('correction-name').value=notKeeper?'':p.identity_correction?.name||'';
  $('correction-note').value=notKeeper?'The highlighted player is not the goalkeeper.':p.identity_correction?.note||'';
  $('correction-reviewer').value=safeLocalGet('pitchprofile-labeller')||$('mp-labeler').value||'';
  $('identify-reviewer-details').open=!$('correction-reviewer').value.trim();
  $('identify-reviewer-label').textContent=$('correction-reviewer').value.trim()?'Reviewing as '+$('correction-reviewer').value.trim()+' · change':'Your name · enter once';
  $('identify-extra').open=false;$('correction-status').textContent='';
  $('correction-reset').hidden=!p.identity_correction&&p.identity_status!=='confirmed';
  $('identify-current').textContent='Currently labelled: '+p.name;
  setIdentityRole(notKeeper?'player':String(p.role).toLowerCase()==='goalkeeper'?'goalkeeper':'player');
  setIdentityMode(notKeeper||!identityTeams().length?'quick':'roster');renderIdentityTeams();showIdentityFrame();
  $('identify-roster-back').textContent=identityTeams().length?'← Choose from the line-up':'Find the match line-up';
  if(!$('identify-dialog').open)$('identify-dialog').showModal();
  (identityPicker.mode==='quick'?document.querySelector('[data-identity-role="player"]'):identityTeam()?$('identify-search'):$('identify-teams').querySelector('button'))?.focus();
}
function closeIdentityPicker(){if(!identityPicker.busy)$('identify-dialog').close();}
async function submitIdentity(reset=false){
  if(identityPicker.busy||identityPicker.key!==reviewDesk.key)return;
  if(!$('correction-reviewer').value.trim()){$('identify-reviewer-details').open=true;$('correction-reviewer').focus();$('correction-status').textContent='Enter your name once so this correction can be attributed to you.';return;}
  if(!reset&&!$('identify-form').reportValidity())return;
  const candidate=identityCandidate(),quick=identityPicker.mode==='quick';
  if(!reset&&!quick&&!candidate)return;
  const key=reviewDesk.key,id=state.manifest.id,pid=state.pid,reviewer=$('correction-reviewer').value.trim(),root=matchBase()+'/players/'+encodeURIComponent(pid);
  const {start,end}=videoBounds(),video=$('mp-video'),time_s=Math.max(start,Math.min(end-.001,Number(video.dataset.seek??video.currentTime)));
  const payload=quick||reset?{role:$('correction-role').value,jersey:reset?null:$('correction-shirt').value===''?null:Number($('correction-shirt').value),name:reset?'':$('correction-name').value,reviewer,note:reset?'':$('correction-note').value,time_s,reset}:
    {athlete_id:candidate.id,team_id:identityPicker.team,reviewer,note:'Confirmed against the highlighted player in the footage.'};
  const controls=[...$('identify-form').querySelectorAll('button,input')],disabled=controls.map(c=>c.disabled);
  identityPicker.busy=true;controls.forEach(c=>c.disabled=true);$('correction-status').textContent='Saving…';
  let saved=false;
  try{
    await post(root+(quick||reset?'/track-correction':'/lineup-identity'),payload);saved=true;
    safeLocalSet('pitchprofile-labeller',reviewer);
    await refreshCorrectedPlayer(id,pid);
    if(reviewDesk.key===key){$('identify-dialog').close();$('identity-save-status').textContent=reset?'Model identity restored.':quick?'Correction saved.':'✓ '+candidate.name+' confirmed.';}
    notice(reset?'Model identity restored.':quick?'Player correction saved.':candidate.name+' confirmed.');
  }catch(error){if(reviewDesk.key===key)$('correction-status').textContent=(saved?'Saved, but the screen could not refresh. Reopen this player to see it. ':'Could not save. Your choices are kept; try again. ')+error.message;}
  finally{identityPicker.busy=false;controls.forEach((c,i)=>c.disabled=disabled[i]);updateIdentitySelection();}
}

$('mp-correct-player').onclick=()=>openPlayerCorrection();
$('mp-not-goalkeeper').onclick=()=>openPlayerCorrection(true);
for(const id of ['identify-close','identify-cancel','identify-watch'])$(id).onclick=closeIdentityPicker;
$('identify-dialog').addEventListener('cancel',e=>{if(identityPicker.busy)e.preventDefault();});
$('identify-dialog').addEventListener('close',()=>identityPicker.generation++);
$('identify-form').onsubmit=e=>{e.preventDefault();submitIdentity();};
$('correction-reset').onclick=()=>submitIdentity(true);
$('identify-manual').onclick=()=>{identityPicker.generation++;setIdentityMode('quick');};
$('identify-roster-back').onclick=()=>setIdentityMode('roster');
$('identify-search').oninput=()=>{identityPicker.athlete=null;renderIdentityCandidates();updateIdentitySelection();};
$('identify-teams').onclick=e=>{const b=e.target.closest('[data-identity-team]');if(b){identityPicker.team=b.dataset.identityTeam;identityPicker.athlete=null;renderIdentityTeams();$('identify-search').focus();}};
$('identify-candidates').onclick=e=>{const b=e.target.closest('[data-athlete]');if(b){identityPicker.athlete=b.dataset.athlete;renderIdentityCandidates();updateIdentitySelection();}};
document.querySelectorAll('[data-identity-role]').forEach(b=>b.onclick=()=>setIdentityRole(b.dataset.identityRole));
for(const id of ['correction-shirt','correction-name'])$(id).oninput=updateIdentitySelection;
$('correction-reviewer').oninput=()=>{$('identify-reviewer-label').textContent=$('correction-reviewer').value.trim()?'Reviewing as '+$('correction-reviewer').value.trim():'Your name · enter once';};
$('identify-load').onclick=async()=>{
  const key=reviewDesk.key,id=state.manifest.id,pid=state.pid,generation=identityPicker.generation,b=$('identify-load');b.disabled=true;$('correction-status').textContent='Loading the match line-up…';
  try{await post(matchBase()+'/match/context',{});await refreshCorrectedPlayer(id,pid);if(reviewDesk.key===key&&identityPicker.generation===generation){renderMatchContext();setIdentityMode('roster');renderIdentityTeams();$('identify-roster-back').hidden=false;$('correction-status').textContent='';}}
  catch(error){if(reviewDesk.key===key&&identityPicker.generation===generation)$('correction-status').textContent='Line-up unavailable. You can enter a number or role instead. '+error.message;}
  finally{b.disabled=false;}
};
window.addEventListener('profilechanging',()=>{if($('identify-dialog').open)$('identify-dialog').close();identityPicker.key=null;});
