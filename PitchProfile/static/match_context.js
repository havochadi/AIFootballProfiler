'use strict';

function linkedTeamName(kit){
  const info=matchState.data?.match_context;
  return info?.context?.teams.find(t=>t.id===info.binding?.teams?.[kit])?.name||matchState.data?.teams?.[kit]?.name||kit;
}
function teamCrest(t){return t?.logo?`<img src="${esc(t.logo)}" alt="${esc(t.name)} crest" loading="lazy" referrerpolicy="no-referrer">`:'<span class="crest-placeholder" aria-hidden="true">◈</span>';}
function renderMatchContext(){
  const data=currentMatch();if(!data)return;
  const info=data.match_context||{},c=info.context,home=c?.teams.find(t=>t.side==='home'),away=c?.teams.find(t=>t.side==='away');
  $('fixture-banner').innerHTML=`<div class="fixture-kicker"><span>${esc(c?.competition||'MATCH REVIEW')}</span><span class="fixture-status">${esc(c?.status||'VIDEO ANALYSED')}</span></div><div class="fixture-scoreline"><div class="fixture-team">${teamCrest(home)}<strong>${esc(home?.name||data.fixture_teams?.home||'Home team')}</strong></div><div class="fixture-score">${c?`${esc(home?.score??'—')} <span>:</span> ${esc(away?.score??'—')}`:'<span>VS</span>'}<small>${c?'FINAL SCORE':'CONNECT MATCH DATA'}</small></div><div class="fixture-team away"><strong>${esc(away?.name||data.fixture_teams?.away||'Away team')}</strong>${teamCrest(away)}</div></div><div class="fixture-footer"><span>${esc(state.manifest?.date||'')} · HALF ${state.manifest?.half||1}${c?.venue?' · '+esc(c.venue):''}</span>${c?`<a href="${esc(c.source_url)}" target="_blank" rel="noopener noreferrer">ESPN match source ↗</a>`:'<button type="button" id="banner-connect">Get line-ups & score ↗</button>'}</div>`;
  if($('banner-connect'))$('banner-connect').onclick=()=>{$('match-context-panel').open=true;$('context-connect').focus();};
  $('context-connect').textContent=c?'Refresh match data ↻':'Connect match data ↗';
  $('context-status').textContent=c?`Historical match ${c.event_id} · ${c.teams.reduce((n,t)=>n+t.players.length,0)} squad members · fetched ${new Date(c.fetched).toLocaleDateString()}. ${info.binding?.teams?'Kit groups confirmed.':'Confirm the kits below to identify players.'}`:'Connect a source to see the match-day squads, final score and available portraits.';
  $('kit-binding').hidden=!c;
  $('team-names').hidden=!!info.binding?.teams;
  if(!c){$('official-lineups').innerHTML='';return;}
  for(const kit of ['A','B']){
    $('binding-swatch-'+kit.toLowerCase()).style.background=data.teams?.[kit]?.colour||'#5d738b';
    const select=$('binding-'+kit.toLowerCase());select.innerHTML='<option value="">Choose the team wearing this kit</option>'+c.teams.map(t=>`<option value="${esc(t.id)}">${esc(t.name)}</option>`).join('');
    select.value=info.binding?.teams?.[kit]||'';
  }
  $('binding-reviewer').value=info.binding?.reviewer||safeLocalGet('pitchprofile-labeller')||'';
  $('official-lineups').innerHTML=c.teams.map(t=>`<section class="lineup-team"><div class="heading-row"><h3>${teamCrest(t)}${esc(t.name)}</h3><span class="badge">${t.side==='home'?'HOME':'AWAY'}</span></div>${[true,false].map(start=>`<h4>${start?'Starting XI':'Substitutes'}</h4><div>${t.players.filter(p=>p.starter===start).map(p=>`<div class="lineup-row"><span class="lineup-number">${p.shirt??'—'}</span><a href="${esc(p.profile_url)}" target="_blank" rel="noopener noreferrer">${esc(p.name)}</a><small>${start?esc(p.position||''):p.played?'Came on':'Unused'}</small></div>`).join('')}</div>`).join('')}</section>`).join('');
}

function renderPlayerIdentity(){
  const data=currentMatch(),p=data?.players.find(p=>p.identity===state.pid);if(!p)return;
  const person=p.lineup_player,teamName=p.team_name||linkedTeamName(p.team);
  $('mp-team').textContent=teamName+(p.jersey!=null?' · #'+p.jersey:'')+' · '+(person?.position|| (p.role==='goalkeeper'?'Goalkeeper':'Outfield player'));
  $('mp-name').textContent=p.name;
  $('mp-portrait').innerHTML=`<span class="portrait-number">${p.jersey??'?'}</span>${p.headshot?`<img src="${esc(p.headshot)}" alt="${esc(p.name)} portrait from ESPN" referrerpolicy="no-referrer">`:''}`;
  const statuses={corrected:'✓ Player corrected by '+p.identity_reviewer,confirmed:'✓ Identity confirmed by '+p.identity_reviewer,lineup_match:'Line-up match · verify the detected shirt in the footage',unused_substitute:'⚠ This number belongs to an unused substitute — check the identity',not_in_lineup:'⚠ This number is not in the match-day squad — correct it below',kit_unmapped:'Name not confirmed · choose Identify player',no_context:'Name unknown · you can correct the player directly from the footage'};
  $('mp-identity-status').textContent=statuses[p.identity_status]||'';
  $('mp-identity-status').dataset.state=p.identity_status;
  $('mp-sub').textContent=`${clock(p.visible_seconds)} on screen · track ${p.identity}${p.automatic_jersey!=null&&p.automatic_jersey!==p.jersey?' · model read #'+p.automatic_jersey:''}`;
  $('mp-not-goalkeeper').hidden=String(p.role).toLowerCase()!=='goalkeeper';
  $('mp-correct-player').textContent=['confirmed','corrected'].includes(p.identity_status)?'Change player':'Identify player';
  $('identity-save-status').textContent=['confirmed','corrected'].includes(p.identity_status)?'Identity saved':'Check the player in the yellow box';
}

async function refreshCorrectedPlayer(id,pid){
  const root='/api/datasets/'+encodeURIComponent(id);
  const [match,profile,dataset]=await Promise.all([api(root+'/match'),api(root+'/players/'+encodeURIComponent(pid)),api(root)]);
  if(state.manifest?.id!==id||state.pid!==pid)return;
  matchState.data=match;state.player=profile;state.profiles=dataset.profiles;
  renderMatchPlayers();renderPlayers();renderPlayerIdentity();
  if(eventMap.data)renderEventMap();
}

async function refreshConnectedPlayer(id,pid){
  if(state.manifest?.id!==id)return;
  matchState.data=null;await selectDataset(id,pid);
}
$('context-connect').onclick=handler(async()=>{
  if(!isMatch())return;
  const id=state.manifest.id,pid=state.pid,b=$('context-connect');b.disabled=true;$('context-status').textContent='Finding the historical fixture and line-ups…';
  try{
    await post(matchBase()+'/match/context',{event:$('context-event').value.trim()||null,league:$('context-league').value||null,refresh:true});
    await refreshConnectedPlayer(id,pid);notice('Match data connected. Confirm kit A and B against the footage.');
  }catch(e){if(state.manifest?.id===id)$('context-status').textContent=e.message;throw e;}finally{b.disabled=false;}
});
$('kit-binding').onsubmit=handler(async e=>{
  e.preventDefault();const id=state.manifest.id,pid=state.pid;
  const r=await post(matchBase()+'/match/kit-binding',{teams:{A:$('binding-a').value,B:$('binding-b').value},reviewer:$('binding-reviewer').value});
  safeLocalSet('pitchprofile-labeller',$('binding-reviewer').value);$('match-context-panel').open=false;
  if(r.job_id){notice('Kit groups confirmed. Naming players with this match’s line-up…');await monitor(r.job_id,id);if(state.manifest?.id===id)await loadMatch();}
  await refreshConnectedPlayer(id,pid);notice('Kit groups confirmed. Shirt numbers are now checked against this match’s line-up.');
});
window.addEventListener('reviewready',renderPlayerIdentity);
document.addEventListener('error',e=>{if(e.target.tagName==='IMG'){e.target.hidden=true;if(e.target.closest('#mp-portrait'))e.target.parentElement.title='Portrait unavailable; shirt number shown instead.';}},true);
