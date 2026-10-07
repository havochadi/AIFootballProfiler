'use strict';
const eventMap={key:null,data:null,request:0,view:'video',kind:'pass',outcome:'',selected:null};
const eventNames={pass:'Passes',shot:'Shots',cross:'Crosses',high_pass:'Lofted passes',carry:'Carries',take_on:'Take-ons',tackle:'Tackles',interception:'Interceptions',block:'Blocks',header:'Headers',recovery:'Recoveries',clearance:'Clearances',pressure:'Pressures',touch:'Touches'};
const panelKinds=['profile','movement'];
const outcomeNames={complete:'Complete',intercepted:'Intercepted',incomplete:'Incomplete',out_of_play:'Out of play',goal:'Goal',on_target:'On target / saved',off_target:'Off target',blocked:'Blocked',woodwork:'Woodwork',unknown:'Unknown',won:'Won',lost:'Lost',successful:'Successful',unsuccessful:'Unsuccessful',cleared:'Cleared',effective:'Effective',ineffective:'Ineffective',observed:'Observed'};
const outcomeColours={complete:'#69e1c8',intercepted:'#ff7c80',incomplete:'#ff7c80',out_of_play:'#ffac7c',goal:'#c8f55b',on_target:'#68c9ff',off_target:'#ff7c80',blocked:'#ffca73',woodwork:'#bf9bff',unknown:'#8f9cb1',won:'#69e1c8',lost:'#ff7c80',successful:'#69e1c8',unsuccessful:'#ff7c80',cleared:'#68c9ff',effective:'#69e1c8',ineffective:'#ff7c80',observed:'#68c9ff'};
function chooseEvidenceView(view){
  eventMap.view=view;$('video-evidence').hidden=view!=='video';$('event-evidence').hidden=view!=='map';
  $('event-inspector').hidden=view!=='map'||panelKinds.includes(eventMap.kind)||!eventMap.selected;
  document.querySelectorAll('[data-evidence-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.evidenceView===view)));
  if(view==='map'){$('mp-video').pause();handler(loadEventMap)();}
}
async function loadEventMap(){
  if(!reviewDesk.ready||!currentMatch())return;
  const key=reviewDesk.key;if(eventMap.key===key&&eventMap.data){renderEventMap();return;}
  const request=++eventMap.request;
  $('event-map-note').textContent='Loading pitch events…';$('event-pitch').innerHTML='';$('map-event-list').innerHTML='';
  const data=await api(matchBase()+'/match/event-map?player='+encodeURIComponent(state.pid));
  if(request!==eventMap.request||reviewDesk.key!==key)return;
  eventMap.key=key;eventMap.data=data;eventMap.outcome='';eventMap.selected=null;
  const {start,end}=videoBounds();$('event-map-from').value=(start/60).toFixed(2);$('event-map-to').value=(end/60).toFixed(2);
  renderEventMap();
}
function eventPosition(e){
  if(e.x==null||e.y==null)return null;
  if($('event-map-direction').value==='stadium')return {x:e.x,y:e.y,ex:e.end_x,ey:e.end_y};
  if(!e.attack_sign)return null;
  return e.attack_sign<0?{x:105-e.x,y:68-e.y,ex:e.end_x==null?null:105-e.end_x,ey:e.end_y==null?null:68-e.end_y}:{x:e.x,y:e.y,ex:e.end_x,ey:e.end_y};
}
function filteredMapEvents(ignoreOutcome=false){
  const from=Number($('event-map-from').value)*60,to=Number($('event-map-to').value)*60;
  return (eventMap.data?.events||[]).filter(e=>e.type===eventMap.kind&&e.time_s>=from&&e.time_s<=to&&
    (ignoreOutcome||!eventMap.outcome||e.outcome===eventMap.outcome));
}
function pitchLines(){
  return '<g fill="none" stroke="#8ea8bd" stroke-opacity=".42" stroke-width="2"><rect x="40" y="40" width="1050" height="680" rx="2"/><path d="M565 40v680M40 178.4h165v403.2H40m1050-403.2H925v403.2h165M40 288.4h55v183.2H40m1050-183.2h-55v183.2h55"/><circle cx="565" cy="380" r="91.5"/><path d="M205 306a91.5 91.5 0 0 1 0 148m720-148a91.5 91.5 0 0 0 0 148"/><circle cx="150" cy="380" r="3" fill="#8ea8bd"/><circle cx="980" cy="380" r="3" fill="#8ea8bd"/></g>';
}
function renderEventMap(){
  const data=eventMap.data;if(!data)return;
  $('event-kind-buttons').innerHTML=`<button type="button" data-kind="profile" aria-pressed="${eventMap.kind==='profile'}">Percentile profile</button><button type="button" data-kind="movement" aria-pressed="${eventMap.kind==='movement'}">Movement</button>`+Object.entries(eventNames).map(([k,n])=>`<button type="button" data-kind="${k}" aria-pressed="${eventMap.kind===k}">${n}<span>${data.events.filter(e=>e.type===k).length}</span></button>`).join('');
  $('movement-evidence').hidden=eventMap.kind!=='movement';$('profile-evidence').hidden=eventMap.kind!=='profile';
  $('event-map-content').hidden=panelKinds.includes(eventMap.kind);
  if(eventMap.kind==='profile')$('mp-profile').innerHTML=styleProfile(currentMatch()?.players.find(p=>p.identity===state.pid)||{});
  if(panelKinds.includes(eventMap.kind)){$('event-inspector').hidden=true;return;}
  $('stat-summary').innerHTML=eventStatSummary(currentMatch()?.players.find(p=>p.identity===state.pid)||{},eventMap.kind);
  const all=filteredMapEvents(true),rows=filteredMapEvents(),mapped=rows.filter(e=>eventPosition(e)),unknown=all.filter(e=>e.outcome==='unknown').length,reviewed=all.filter(e=>e.outcome_source==='reviewed').length;
  const counts=Object.fromEntries((data.outcomes[eventMap.kind]||[]).map(k=>[k,all.filter(e=>e.outcome===k).length]));
  $('event-map-totals').innerHTML=`<div><strong>${all.length}</strong><span>${eventNames[eventMap.kind]} in interval</span></div><div><strong>${reviewed}</strong><span>Outcomes reviewed</span></div><div><strong>${unknown}</strong><span>Outcome unknown</span></div>`;
  $('event-map-legend').innerHTML=`<button type="button" data-outcome="" aria-pressed="${!eventMap.outcome}">All <b>${all.length}</b></button>`+Object.entries(counts).map(([k,n])=>`<button type="button" data-outcome="${k}" aria-pressed="${eventMap.outcome===k}"><i style="background:${outcomeColours[k]}"></i>${outcomeNames[k]} <b>${n}</b></button>`).join('');
  const heat=$('event-map-mode').value==='heatmap';
  let plot='<defs><linearGradient id="pitch-shade" x2="1" y2="1"><stop stop-color="#142f3b"/><stop offset="1" stop-color="#10222f"/></linearGradient><pattern id="pitch-grid" width="105" height="68" patternUnits="userSpaceOnUse"><path d="M105 0H0V68" fill="none" stroke="#8fbbce" stroke-opacity=".07"/></pattern></defs><rect x="40" y="40" width="1050" height="680" fill="url(#pitch-shade)"/><rect x="40" y="40" width="1050" height="680" fill="url(#pitch-grid)"/>';
  if(heat){
    const bins=new Map();
    for(const e of mapped){const p=eventPosition(e),x=Math.min(20,Math.floor(p.x/5)),y=Math.min(13,Math.floor(p.y/(68/14))),key=x+':'+y;if(!bins.has(key))bins.set(key,{x,y,events:[]});bins.get(key).events.push(e);}
    const max=Math.max(1,...[...bins.values()].map(b=>b.events.length));
    for(const bin of bins.values()){
      const t=bin.events.length/max,colour=t>.66?'#d4f75d':t>.33?'#64d9a4':'#379bb2';
      plot+=`<g role="button" tabindex="0" data-map-event="${bin.events[0].id}" aria-label="${bin.events.length} events in this area; inspect first at ${clock(bin.events[0].time_s)}"><rect x="${40+bin.x*50}" y="${40+bin.y*680/14}" width="50" height="${680/14}" fill="${colour}" fill-opacity="${.2+.7*Math.sqrt(t)}"/><title>${bin.events.length} events · select to inspect</title></g>`;
    }
    plot+=pitchLines();
  }else{
    plot+=pitchLines();
    for(const e of mapped.slice(0,1500)){
      const p=eventPosition(e),x=40+p.x*10,y=40+p.y*10,colour=outcomeColours[e.outcome]||'#8f9cb1';
      if(p.ex!=null&&p.ey!=null&&['pass','carry','dribble'].includes(e.type)){
        const ex=40+p.ex*10,ey=40+p.ey*10,a=Math.atan2(ey-y,ex-x),size=7;
        plot+=`<path d="M${x} ${y}L${ex} ${ey}m${-size*Math.cos(a-.5)} ${-size*Math.sin(a-.5)}L${ex} ${ey}l${-size*Math.cos(a+.5)} ${-size*Math.sin(a+.5)}" fill="none" stroke="${colour}" stroke-width="2" opacity=".5" pointer-events="none"/>`;
      }
      plot+=`<g role="button" tabindex="0" data-map-event="${e.id}" aria-label="${esc(eventNames[e.type])} at ${clock(e.time_s)}, ${esc(outcomeNames[e.outcome])}"><circle cx="${x}" cy="${y}" r="${e.id===eventMap.selected?13:8}" fill="${e.outcome==='unknown'?'#192d40':colour}" stroke="${e.id===eventMap.selected?'#fff':colour}" stroke-width="3"/><title>${clock(e.time_s)} · ${outcomeNames[e.outcome]} · ${e.outcome_source==='reviewed'?'reviewed':'model'}</title></g>`;
    }
  }
  plot+=`<text x="565" y="23" fill="#a6bdcc" font-size="15" text-anchor="middle" letter-spacing="3">${$('event-map-direction').value==='attack'?'ATTACKING DIRECTION →':'STADIUM COORDINATES'}</text><text x="565" y="749" fill="#829aaf" font-size="14" text-anchor="middle">${heat?'EVENT ORIGINS · BRIGHTER = MORE EVENTS':'SELECT A MARKER TO REVIEW THE MOMENT'}</text>`;
  $('event-pitch').innerHTML=plot;
  $('event-map-empty').hidden=!!mapped.length;$('event-map-empty').textContent=!all.length?'No '+eventNames[eventMap.kind].toLowerCase()+' detected in this interval.':!rows.length?'No events with this outcome.':'No usable positions for these events. Try stadium view or inspect the list.';
  $('event-map-note').textContent=`${mapped.length} of ${rows.length} filtered events have usable positions${$('event-map-direction').value==='attack'?' and a known attack direction':''}. `+(eventMap.kind==='shot'?'The model detects attempts; goal, saved and blocked outcomes require review. ':['tackle','interception'].includes(eventMap.kind)?'Model detections cover successful possession changes, not all attempted challenges. ':'')+(heat?'Heatmap counts event origins within the selected filters. ':mapped.length>1500?'First 1,500 markers shown; use the heatmap for all events. ':'')+'Source: video analysis and your saved outcome reviews.';
  $('map-event-list').innerHTML=rows.length?rows.slice(0,100).map(e=>`<button type="button" data-map-event="${e.id}" class="map-event-row ${eventMap.selected===e.id?'selected':''}"><span>${clock(e.time_s)}</span><strong>${outcomeNames[e.outcome]}</strong><small>${e.outcome_source==='reviewed'?'✓ Reviewed':e.outcome_source==='unclassified'?'Needs review':'Model'}${eventPosition(e)?'':' · no map position'}</small></button>`).join('')+(rows.length>100?'<p class="small muted">First 100 events listed; narrow the time range to see later events.</p>':''):'<p class="small muted">No events match these filters.</p>';
}

function selectMappedEvent(id){
  const e=eventMap.data?.events.find(e=>e.id===id);if(!e)return;
  eventMap.selected=id;renderEventMap();$('event-inspector').hidden=false;
  $('selected-event-title').textContent=eventNames[e.type]+' · '+clock(e.time_s)+' · '+outcomeNames[e.outcome];
  $('selected-event-description').textContent=(e.outcome_source==='reviewed'?'Reviewed by '+e.review.reviewer:e.outcome_source==='unclassified'?'Outcome not classified by the detector':'Outcome inferred by the video model')+(e.length_m!=null?' · '+num(e.length_m)+' m':'')+'. Watch the source moment before confirming an outcome.';
  $('selected-event-outcome').innerHTML=eventMap.data.outcomes[e.type].map(k=>`<option value="${k}">${outcomeNames[k]}</option>`).join('');$('selected-event-outcome').value=e.outcome;
  $('selected-event-reviewer').value=e.review?.reviewer||safeLocalGet('pitchprofile-labeller')||'';
  $('selected-event-note').value=e.review?.notes||'';$('event-outcome-status').textContent='';
  const p=currentMatch()?.players.find(p=>p.identity===state.pid),c=currentMatch()?.match_context?.context;
  const normalized=s=>String(s||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
  const reference=c?.reported_shots.filter(r=>normalized(r.player)===normalized(p?.lineup_player?.name)&&r.period===(state.manifest.half||1))||[];
  $('reported-player-shots').innerHTML=reference.length?reference.map(r=>`<div class="reference-shot"><strong>${esc(r.clock)} · ${esc(outcomeNames[r.outcome]||r.outcome)}</strong><p>${esc(r.description)}</p>${e.type==='shot'&&eventMap.data.outcomes.shot.includes(r.outcome)?`<button type="button" data-reference-outcome="${r.outcome}" data-reference-id="${esc(r.id)}" data-reference-clock="${esc(r.clock)}">Use this outcome after checking footage</button>`:''}</div>`).join(''):'<p class="small muted">No matching reported shots for this player and half. Connect the match and confirm the player identity to use this reference.</p>';
  $('event-inspector').scrollIntoView({block:'nearest',behavior:'smooth'});
}
function resetEventTime(){const {start,end}=videoBounds();$('event-map-from').value=(start/60).toFixed(2);$('event-map-to').value=(end/60).toFixed(2);renderEventMap();}
document.querySelectorAll('[data-evidence-view]').forEach(b=>b.onclick=()=>chooseEvidenceView(b.dataset.evidenceView));
$('event-kind-buttons').onclick=e=>{const b=e.target.closest('[data-kind]');if(b){eventMap.kind=b.dataset.kind;eventMap.outcome='';eventMap.selected=null;$('event-inspector').hidden=true;renderEventMap();}};
$('event-map-legend').onclick=e=>{const b=e.target.closest('[data-outcome]');if(b){eventMap.outcome=b.dataset.outcome;renderEventMap();}};
for(const id of ['event-map-mode','event-map-direction','event-map-from','event-map-to'])$(id).onchange=renderEventMap;
$('event-map-reset').onclick=resetEventTime;
for(const id of ['event-pitch','map-event-list']){
  $(id).onclick=e=>{const el=e.target.closest('[data-map-event]');if(el)selectMappedEvent(el.dataset.mapEvent);};
}
$('event-pitch').onkeydown=e=>{if(['Enter',' '].includes(e.key)){const el=e.target.closest('[data-map-event]');if(el){e.preventDefault();selectMappedEvent(el.dataset.mapEvent);}}};
$('selected-event-watch').onclick=()=>{const e=eventMap.data?.events.find(e=>e.id===eventMap.selected);if(e){chooseEvidenceView('video');seekVideo(e.time_s,true,2);$('mp-video').scrollIntoView({block:'center',behavior:'smooth'});}};
$('event-outcome-form').onsubmit=handler(async ev=>{
  ev.preventDefault();const id=eventMap.selected,key=reviewDesk.key,root=matchBase(),payload={outcome:$('selected-event-outcome').value,reviewer:$('selected-event-reviewer').value,notes:$('selected-event-note').value};
  const button=ev.target.querySelector('button');button.disabled=true;
  try{
    const saved=await post(root+'/match/events/'+id+'/review',payload);
    if(reviewDesk.key!==key||!eventMap.data)return;
    const e=eventMap.data.events.find(e=>e.id===id);if(e){e.outcome=saved.outcome;e.outcome_source='reviewed';e.review=saved;}
    renderEventMap();if(e&&eventMap.selected===id){
      const unchanged=$('selected-event-outcome').value===payload.outcome&&$('selected-event-reviewer').value===payload.reviewer&&$('selected-event-note').value===payload.notes;
      $('event-outcome-status').textContent=unchanged?'Outcome saved':'Previous outcome saved · newer form edits still need saving';
      $('selected-event-title').textContent=eventNames[e.type]+' · '+clock(e.time_s)+' · '+outcomeNames[e.outcome];
      $('selected-event-description').textContent='Reviewed by '+saved.reviewer+(e.length_m!=null?' · '+num(e.length_m)+' m':'')+'. Original model output is retained separately.';
    }
  }finally{button.disabled=false;}
});
$('reported-player-shots').onclick=e=>{const b=e.target.closest('[data-reference-outcome]');if(b){$('selected-event-outcome').value=b.dataset.referenceOutcome;$('selected-event-note').value=`Checked against ESPN match event ${b.dataset.referenceId} (${b.dataset.referenceClock}).`;$('event-outcome-status').textContent='Check the video, then save to confirm this match.';}};
window.addEventListener('profilechanging',()=>{++eventMap.request;eventMap.key=null;eventMap.data=null;eventMap.selected=null;$('event-inspector').hidden=true;});
window.addEventListener('reviewready',()=>{if(eventMap.view==='map')handler(loadEventMap)();});
