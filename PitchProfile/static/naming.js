'use strict';

// Name players: every consistent player without a shirt number, with pictures and the number
// reader's best guess; all typed numbers (or "not a player" marks, for a referee or coach the
// model grouped with a team) are applied in one re-analysis. Names are stored per footage
// segment on the server, so they survive later re-analysis.
const NOT_PLAYER='none';
const naming={numbers:{},crops:{},loading:null,busy:false};
const namingValid=v=>v===NOT_PLAYER||Number.isInteger(v)&&v>=1&&v<=99||typeof v==='string'&&v.startsWith('same:');
const sameTarget=v=>typeof v==='string'&&v.startsWith('same:')?v.slice(5):null;

const namingTeamName=(data,key)=>data.teams?.[key]?.name||('Team '+key);
function unnamedPlayers(){
  const data=currentMatch(),team=$('naming-team').value;
  return data?data.players.filter(p=>p.unnamed&&(!team||p.team===team)).sort((a,b)=>(b.visible_seconds||0)-(a.visible_seconds||0)):[];
}
function namingGuess(p){
  if(p.number_guess==null||!p.readable_views)return 'No readable shirt number in its pictures';
  const share=Math.round((p.number_guess_share||0)*100);
  return share>=30?`Reader's best guess: #${p.number_guess} (${share}% of ${p.readable_views} readable views)`:`No clear reading (${p.readable_views} readable views)`;
}
function namingJoin(p,n){
  const data=currentMatch();if(!data||!n)return '';
  if(n===NOT_PLAYER)return 'Removed from player statistics (a referee, coach or other non-player).';
  const same=sameTarget(n);
  if(same){const q=data.players.find(x=>x.identity===same);return q?`Same person as ${q.name}: their statistics are combined.`:'';}
  const named=data.players.find(q=>q.team===p.team&&!q.unnamed&&q.jersey===n);
  const twin=Object.entries(naming.numbers).some(([pid,v])=>pid!==p.identity&&v===n&&data.players.find(q=>q.identity===pid)?.team===p.team);
  if(named)return `Joins ${named.name} (${clock(named.visible_seconds)} on screen): their statistics are combined.`;
  if(twin)return `Another player here also gets #${n}: they will be combined.`;
  return `New player: ${namingTeamName(data,p.team)} #${n}.`;
}
function renderNaming(){
  const data=currentMatch(),panel=$('naming-panel');
  panel.hidden=!data;if(!data)return;
  if(naming.shown!==data.id){naming.shown=data.id;naming.numbers={};naming.crops={};}
  const all=data.players.filter(p=>p.unnamed),team=$('naming-team').value;
  $('naming-team').innerHTML='<option value="">Both teams</option>'+Object.keys(data.teams||{}).map(k=>`<option value="${esc(k)}">${esc(namingTeamName(data,k))}</option>`).join('');
  $('naming-team').value=team in (data.teams||{})?team:'';
  const time=all.reduce((s,p)=>s+(p.visible_seconds||0),0),total=data.players.reduce((s,p)=>s+(p.visible_seconds||0),0)||1;
  $('naming-count').textContent=all.length?`${all.length} unnamed · ${Math.round(time/total*100)}% of player time`:'everyone has a name';
  const rows=unnamedPlayers();
  $('naming-list').innerHTML=rows.length?rows.map((p,i)=>{
    const v=naming.numbers[p.identity],off=v===NOT_PLAYER,same=sameTarget(v),n=off||same||v==null?'':v,guess=p.number_guess!=null&&(p.number_guess_share||0)>=.3;
    const others=data.players.filter(q=>q.team===p.team&&q.identity!==p.identity).sort((a,b)=>(a.unnamed-b.unnamed)||((a.jersey??100)-(b.jersey??100))||((b.visible_seconds||0)-(a.visible_seconds||0)));
    return `<div class="naming-row" data-pid="${esc(p.identity)}">
      <div class="naming-crops" data-crops="${esc(p.identity)}" aria-label="Pictures of ${esc(p.name)}">${cropMarkup(p.identity)}</div>
      <div class="naming-info"><strong>${esc(p.name)}</strong><span>${clock(p.visible_seconds)} on screen</span><span class="naming-guess">${esc(namingGuess(p))}</span></div>
      <div class="naming-entry">
        <label for="naming-n-${i}">Shirt number<input id="naming-n-${i}" type="number" min="1" max="99" step="1" inputmode="numeric" data-number="${esc(p.identity)}" value="${esc(n)}"${off||same?' disabled':''}></label>
        <label for="naming-s-${i}">Or the same person as<select id="naming-s-${i}" data-same="${esc(p.identity)}"${off?' disabled':''}><option value="">Choose a player…</option>${others.map(q=>`<option value="${esc(q.identity)}"${q.identity===same?' selected':''}>${esc(q.name)} · ${clock(q.visible_seconds)}</option>`).join('')}</select></label>
        <div class="naming-buttons">${guess?`<button type="button" data-use="${esc(p.identity)}" data-guess="${p.number_guess}">Use #${p.number_guess}</button>`:''}<button type="button" data-watch="${esc(p.identity)}">Watch</button><button type="button" data-notplayer="${esc(p.identity)}" aria-pressed="${off}">Not a player</button></div>
        <small class="naming-join" data-join="${esc(p.identity)}">${esc(namingJoin(p,off?NOT_PLAYER:same?v:n===''?null:Number(n)))}</small>
      </div></div>`;}).join(''):'<p class="empty">No unnamed players on this team.</p>';
  const given=data.players.filter(p=>p.identity_confirmed),removed=data.not_player_segments||0;
  $('naming-given').hidden=!given.length&&!removed;
  $('naming-given-list').innerHTML=given.map(p=>`<div class="naming-given-row"><span>${esc(p.name)} · ${clock(p.visible_seconds)} on screen</span><button type="button" data-undo="${esc(p.identity)}">Undo my name</button></div>`).join('')+
    (removed?`<div class="naming-given-row"><span>${removed} ${removed===1?'piece':'pieces'} of footage marked as not a player</span><button type="button" data-undo="NONE">Restore as players</button></div>`:'');
  updateNamingApply();
  if(panel.open)loadNamingCrops();
}
function cropMarkup(pid){
  const urls=naming.crops[pid];
  if(!urls)return `<span class="naming-crop-wait">${$('naming-panel').open?'Loading pictures…':'Pictures load when this list is open'}</span>`;
  return urls.length?urls.map(u=>`<img src="${esc(u)}" alt="" loading="lazy">`).join(''):'<span class="naming-crop-wait">No pictures kept for this player</span>';
}
async function loadNamingCrops(){
  const data=currentMatch();if(!data||naming.loading===data.id)return;
  naming.loading=data.id;
  try{
    for(const p of unnamedPlayers()){
      if(currentMatch()?.id!==data.id)return;
      if(naming.crops[p.identity])continue;
      try{naming.crops[p.identity]=(await api(matchBase()+'/match/crops/'+encodeURIComponent(p.identity)+'?limit=6')).crops;}
      catch{naming.crops[p.identity]=[];}
      const box=document.querySelector(`[data-crops="${CSS.escape(p.identity)}"]`);if(box)box.innerHTML=cropMarkup(p.identity);
    }
  }finally{if(naming.loading===data.id)naming.loading=null;}
}
function updateNamingApply(){
  const n=Object.values(naming.numbers).filter(namingValid).length;
  $('naming-apply').disabled=naming.busy||!n;
  $('naming-apply').textContent=n?`Apply ${n} ${n===1?'change':'changes'}`:'Apply names';
}
function setNamingNumber(pid,value){
  const n=value===NOT_PLAYER||sameTarget(value)?value:value===''||value==null?null:Number(value);
  if(!namingValid(n))delete naming.numbers[pid];else naming.numbers[pid]=n;
  document.querySelectorAll('[data-join]').forEach(el=>{const q=currentMatch()?.players.find(x=>x.identity===el.dataset.join);if(q)el.textContent=namingJoin(q,naming.numbers[q.identity]??null);});
  updateNamingApply();
}
async function applyNames(){
  if(naming.busy)return;
  const names=Object.entries(naming.numbers).map(([source,v])=>v===NOT_PLAYER?{source,not_player:true}:sameTarget(v)?{source,same_as:sameTarget(v)}:{source,number:v});if(!names.length)return;
  const id=state.manifest.id;naming.busy=true;updateNamingApply();$('naming-status').textContent='Saving names and re-analysing this half…';
  try{
    const r=await post(matchBase()+'/match/identities/names',{names});
    await monitor(r.job_id,id);
    naming.numbers={};naming.crops={};
    if(state.manifest?.id===id)await loadMatch();
    $('naming-status').textContent=`${names.length} ${names.length===1?'change':'changes'} applied.`;
  }catch(error){$('naming-status').textContent='Could not apply the names. Your numbers are kept; try again. '+error.message;throw error;}
  finally{naming.busy=false;renderNaming();}
}
async function undoName(identity){
  if(naming.busy)return;
  const id=state.manifest.id;naming.busy=true;updateNamingApply();$('naming-status').textContent='Removing your name and re-analysing this half…';
  try{
    const r=await post(matchBase()+'/match/identities/undo',{identity});
    await monitor(r.job_id,id);naming.crops={};
    if(state.manifest?.id===id)await loadMatch();
    $('naming-status').textContent='Name removed; the model\'s naming is back.';
  }catch(error){$('naming-status').textContent='Could not undo the name. '+error.message;throw error;}
  finally{naming.busy=false;renderNaming();}
}

$('naming-panel').addEventListener('toggle',()=>{if($('naming-panel').open)loadNamingCrops();});
$('naming-team').onchange=()=>renderNaming();
$('naming-list').addEventListener('input',e=>{const pid=e.target.dataset?.number;if(pid)setNamingNumber(pid,e.target.value);});
$('naming-list').addEventListener('change',e=>{
  const pid=e.target.dataset?.same;if(!pid)return;
  const input=document.querySelector(`[data-number="${CSS.escape(pid)}"]`);
  setNamingNumber(pid,e.target.value?'same:'+e.target.value:input.value);input.disabled=!!e.target.value;
});
$('naming-list').addEventListener('click',handler(async e=>{
  const use=e.target.closest('[data-use]'),watch=e.target.closest('[data-watch]'),off=e.target.closest('[data-notplayer]');
  if(off){const pid=off.dataset.notplayer,input=document.querySelector(`[data-number="${CSS.escape(pid)}"]`),on=naming.numbers[pid]!==NOT_PLAYER;
    setNamingNumber(pid,on?NOT_PLAYER:input.value);off.setAttribute('aria-pressed',String(on));input.disabled=on;}
  if(use){const input=document.querySelector(`[data-number="${CSS.escape(use.dataset.use)}"]`);input.value=use.dataset.guess;setNamingNumber(use.dataset.use,use.dataset.guess);input.focus();}
  if(watch){await selectPlayer(watch.dataset.watch);$('match-player').scrollIntoView({block:'start'});}
}));
$('naming-given-list').addEventListener('click',handler(async e=>{const b=e.target.closest('[data-undo]');if(b)await undoName(b.dataset.undo);}));
$('naming-apply').onclick=handler(applyNames);
