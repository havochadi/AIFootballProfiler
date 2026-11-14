'use strict';
// Live model view (Group D): under the match video, which models ran and what they said near the
// playhead. Each group adds its own row with LiveModels.addRow({id, title, order, render(ctx)}):
// render returns HTML for the moment ctx.time (see README, "Live model view"). Read-only.
const liveModels={key:null,data:null,error:null,request:0,rows:[],playerOnly:false,low:false,mounted:false};
const LIVE_NAMES={pass:'Pass',high_pass:'Lofted pass',cross:'Cross',shot:'Shot',header:'Header',tackle:'Tackle',block:'Block',
  throw_in:'Throw-in',carry:'Carry',dribble:'Dribble',take_on:'Take-on',interception:'Interception',recovery:'Recovery',
  clearance:'Clearance',dispossessed:'Dispossessed',deflection:'Deflection',free_kick:'Free kick'};
const LIVE_CREDIT={picture:'the ball was inside his picture box',pitch:'he was nearest the ball on the pitch (within 3 m)',
  touchline:'he was the only teammate on the touchline (throw-in rule)',possession:'he had the ball'};
const LIVE_NO_CREDIT={'no ball':'the ball was not seen at that moment','ball away from players':'the ball was seen, but no player of that team was near it',
  'team not on screen':'no player of that team was on screen','no team':'the team is unknown (attack direction not found)'};
const LIVE_BEFORE_S=4,LIVE_AFTER_S=1.5,LIVE_NOW_S=.6;

const LiveModels={
  addRow(row){liveModels.rows=liveModels.rows.filter(r=>r.id!==row.id).concat(row).sort((a,b)=>(a.order??50)-(b.order??50));renderLiveModels();},
  get data(){return liveModels.data;},
  refresh(){renderLiveModels();},
};
window.LiveModels=LiveModels;

function liveMount(){
  if(liveModels.mounted)return true;
  const anchor=document.getElementById('mp-visibility');if(!anchor)return false;   // under the appearance timeline
  const link=document.createElement('link');link.rel='stylesheet';link.href='/static/event_live.css';document.head.appendChild(link);
  const box=document.createElement('section');box.id='live-models';box.className='live-models';box.setAttribute('aria-label','Live model view');
  box.innerHTML=`<div class="lm-head"><div><p class="eyebrow">Live model view</p><h3>What the models see at <span id="lm-clock">0:00</span></h3></div>
    <div class="lm-toggles"><label><input type="checkbox" id="lm-player-only"> This player only</label><label><input type="checkbox" id="lm-low"> Low-score spots</label></div></div>
    <div id="lm-models" class="lm-models" aria-label="Models on this match"></div><div id="lm-rows" class="lm-rows"></div><p id="lm-note" class="small muted"></p>`;
  anchor.after(box);
  $('lm-player-only').onchange=e=>{liveModels.playerOnly=e.target.checked;renderLiveModels();};
  $('lm-low').onchange=e=>{liveModels.low=e.target.checked;renderLiveModels();};
  box.addEventListener('click',e=>{const b=e.target.closest('[data-lm-seek]');if(b&&typeof seekVideo==='function')seekVideo(Number(b.dataset.lmSeek),true,1);});
  liveModels.mounted=true;return true;
}

async function loadLiveModels(){
  if(!liveMount())return;
  const match=typeof currentMatch==='function'?currentMatch():null;
  if(!match){liveModels.data=null;renderLiveModels();return;}
  if(!liveStale(match)){renderLiveModels();return;}
  // The match object is replaced whenever it is reloaded (re-analysis, renamed players): fetch again then.
  const request=++liveModels.request;liveModels.key=match.id;liveModels.source=match;liveModels.data=null;liveModels.error=null;
  $('lm-note').textContent='Loading what the models found…';
  try{
    const data=await api('/api/datasets/'+encodeURIComponent(match.id)+'/match/live-events');
    if(request!==liveModels.request)return;
    liveModels.data=data;liveModels.times=data.events.map(e=>e.time_s);
  }catch(error){if(request===liveModels.request){liveModels.error=error.message;renderLiveModels();}return;}
  renderLiveModels();
}

function liveStale(match){return liveModels.key!==match.id||liveModels.source!==match;}
function liveIndex(times,t){let lo=0,hi=times.length;while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<t)lo=mid+1;else hi=mid;}return lo;}
function liveBallSeen(t){
  const spans=liveModels.data?.ball_seen;if(!spans)return null;
  let lo=0,hi=spans.length-1;while(lo<=hi){const mid=(lo+hi)>>1;if(spans[mid][1]<t)lo=mid+1;else if(spans[mid][0]>t)hi=mid-1;else return true;}
  return false;
}
function liveContext(){
  const match=currentMatch(),video=$('mp-video');
  const players=Object.fromEntries((match?.players||[]).map(p=>[p.identity,p]));
  return {time:video.currentTime||0,pid:state.pid,playerOnly:liveModels.playerOnly,low:liveModels.low,match,data:liveModels.data,
    player:id=>players[id]||null,
    playerName:id=>players[id]?.name||id||'—',
    teamName:k=>match?.teams?.[k]?.name||(k?'Team '+k:'Unknown team'),
    seekButton:(t,text)=>`<button type="button" class="lm-seek" data-lm-seek="${Number(t)}">▶ ${esc(text??clock(t))}</button>`};
}
function liveNearby(ctx,before=LIVE_BEFORE_S,after=LIVE_AFTER_S){
  const d=ctx.data;if(!d)return [];
  const times=liveModels.times,out=[];
  for(let i=liveIndex(times,ctx.time-before);i<times.length&&times[i]<=ctx.time+after;i++){
    const e=d.events[i];
    if(e.model==='video'&&e.confident===false&&!ctx.low)continue;
    if(ctx.playerOnly&&e.identity!==ctx.pid)continue;
    out.push(e);
  }
  return out;
}

function liveEventCard(e,ctx){
  const now=Math.abs(e.time_s-ctx.time)<=LIVE_NOW_S,past=e.time_s<ctx.time-LIVE_NOW_S;
  const mine=e.identity&&e.identity===ctx.pid,name=LIVE_NAMES[e.type]||e.type;
  const model=e.model==='video'
    ?`<span class="lm-model video${e.confident===false?' low':''}" title="T-DEED action spotter, trained by Group D. The score behaves like a probability.">AI video model · score ${Math.round((e.score??0)*100)}%${e.confident===false?' · low':''}</span>`
    :'<span class="lm-model rules" title="Rules written by Group D, from where the players and ball are">Event rules</span>';
  const who=e.identity
    ?`<strong>${esc(ctx.playerName(e.identity))}</strong> <span class="muted">${esc(ctx.teamName(e.team))}</span>`
    :`<strong>No player</strong> <span class="muted">${esc(ctx.teamName(e.team))} only</span>`;
  const how=e.identity?(LIVE_CREDIT[e.credited_by]?'Credited because '+LIVE_CREDIT[e.credited_by]:'')
    :(LIVE_NO_CREDIT[e.uncredited]?'Why: '+LIVE_NO_CREDIT[e.uncredited]:'');
  const extra=[];
  if(e.outcome&&e.outcome!=='unknown')extra.push(String(e.outcome).replaceAll('_',' '));
  if(e.receiver_identity)extra.push('to '+ctx.playerName(e.receiver_identity));
  if(e.length_m!=null)extra.push(Math.round(e.length_m)+' m');
  if(e.filled)extra.push('a pass or carry the rules missed');
  if(e.model==='video'&&e.confident===false)extra.push(`counts as ${(e.score??0).toFixed(2)} of a ${name.toLowerCase()} in the stats`);
  return `<li class="lm-event${now?' now':''}${past?' past':''}${mine?' mine':''}">
    <div class="lm-event-top">${ctx.seekButton(e.time_s)}<strong class="lm-kind">${esc(name)}</strong>${model}${now?'<span class="lm-now">now</span>':''}</div>
    <div class="lm-who">${who}${mine?'<span class="lm-mine">selected player</span>':''}</div>
    ${how?`<div class="small muted">${esc(how)}</div>`:''}${extra.length?`<div class="small muted">${esc(extra.join(' · '))}</div>`:''}</li>`;
}

// Group D's row: the events near the playhead, with the model behind each and how its player was picked.
function liveEventsRow(ctx){
  if(!ctx.data)return `<p class="small muted">${liveModels.error?'Could not load what the models found: '+esc(liveModels.error):'Loading…'}</p>`;
  const ball=liveBallSeen(ctx.time),near=liveNearby(ctx);
  const ballLine=ball==null?'':`<p class="lm-ball ${ball?'seen':'unseen'}"><span aria-hidden="true">●</span> Ball ${ball?'seen by the detector':'not seen right now'}${ball?'':' · spotted actions now may get no player'}</p>`;
  if(near.length)return ballLine+'<ul class="lm-events">'+near.map(e=>liveEventCard(e,ctx)).join('')+'</ul>';
  const times=liveModels.times,d=ctx.data;
  let next=null;
  for(let i=liveIndex(times,ctx.time+LIVE_AFTER_S);i<times.length;i++){
    const e=d.events[i];
    if((e.model!=='video'||e.confident!==false||ctx.low)&&(!ctx.playerOnly||e.identity===ctx.pid)){next=e;break;}
  }
  return ballLine+`<p class="small muted">No event in the last ${LIVE_BEFORE_S} seconds${ctx.playerOnly?' for this player':''}.</p>`+
    (next?`<p class="small lm-next">Next: ${esc(LIVE_NAMES[next.type]||next.type)}${next.identity?' by '+esc(ctx.playerName(next.identity)):''} ${ctx.seekButton(Math.max(0,next.time_s-1),clock(next.time_s))}</p>`:'');
}

function liveModelStrip(ctx){
  const near=liveNearby({...ctx,playerOnly:false},LIVE_NOW_S,LIVE_NOW_S),ball=liveBallSeen(ctx.time);
  const busy={action_spotter:near.some(e=>e.model==='video'),event_rules:near.some(e=>e.model==='rules'),detection:ball===true};
  return (ctx.data?.models||[]).map(m=>`<span class="lm-chip${m.ran?'':' off'}${busy[m.id]?' busy':''}" title="${esc(m.detail||'')}"><b>${esc(m.group)}</b>${esc(m.name)}${m.ran?'':' · not run'}</span>`).join('');
}

function renderLiveModels(){
  if(!liveModels.mounted)return;
  const match=typeof currentMatch==='function'?currentMatch():null;
  $('live-models').hidden=!match;
  if(!match)return;
  if(liveStale(match)){loadLiveModels();return;}        // the match finished loading after profilechange
  const ctx=liveContext();
  $('lm-clock').textContent=clock(ctx.time);
  $('lm-models').innerHTML=liveModelStrip(ctx);
  $('lm-rows').innerHTML=liveModels.rows.map(r=>{
    let body;try{body=r.render(ctx);}catch(error){body=`<p class="small muted">This row could not draw: ${esc(error.message)}</p>`;}
    return `<section class="lm-row" data-row="${esc(r.id)}"><h4>${esc(r.title)}</h4>${body}</section>`;
  }).join('');
  $('lm-note').textContent=liveModels.data?.note||'';
}

LiveModels.addRow({id:'events',title:'Events · Group D',order:40,render:liveEventsRow});
for(const name of ['timeupdate','seeked','loadedmetadata'])$('mp-video').addEventListener(name,()=>{if(state.tab==='match')renderLiveModels();});
window.addEventListener('profilechange',handler(loadLiveModels));
window.addEventListener('tabchange',handler(async()=>{if(state.tab==='match')await loadLiveModels();}));
liveMount();
