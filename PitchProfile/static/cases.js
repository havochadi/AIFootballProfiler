'use strict';
const intervalState={catalogue:null,current:null,request:0,reviewRequest:0};
function roleExamples(role){
  if(!role.examples?.length)return '';
  return '<div class="role-examples"><strong>Examples</strong><ul>'+role.examples.map(e=>`<li><span class="example-title">${esc(e.title)}</span>${e.context?` <span class="muted">· ${esc(e.context)}</span>`:''}<p>${esc(e.description)}${e.source&&/^https:\/\//.test(e.source)?` <a href="${esc(e.source)}" target="_blank" rel="noopener noreferrer">Source ↗</a>`:''}</p></li>`).join('')+'</ul></div>';
}
function renderCatalogue(){
  const catalogue=intervalState.catalogue,group=$('catalogue-group').value;
  if(!catalogue)return;
  const groupNames=Object.fromEntries(catalogue.position_groups.map(g=>[g.id,g.name]));
  const roles=catalogue.roles.filter(r=>!group||r.position_groups.includes(group));
  $('catalogue-count').textContent=`${roles.length} of ${catalogue.roles.length} archetypes${group?' · '+groupNames[group]:''} · ${catalogue.position_groups.length} position groups total`;
  $('catalogue-roles').innerHTML=roles.map(r=>`<article class="catalogue-role"><p class="eyebrow">${esc(r.position_groups.map(id=>groupNames[id]||id).join(' · '))}</p><h2>${esc(r.name)}</h2><p>${esc(r.definition)}</p>${roleExamples(r)}<p class="small muted"><strong>Look for:</strong> ${esc(r.evidence_cues)}</p>${r.trainable?'':'<p class="small muted">Descriptive role; excluded from model training.</p>'}</article>`).join('');
}
const catalogueReady=api('/api/taxonomy').then(c=>{
  intervalState.catalogue=c;
  c.roles.forEach(r=>names[r.id]=r.name);
  const options=c.position_groups.map(g=>`<option value="${esc(g.id)}">${esc(g.name)}</option>`).join('');
  $('case-group').innerHTML=options;
  $('identity-group').innerHTML=$('identity-group').firstElementChild.outerHTML+options;
  $('catalogue-group').innerHTML=$('catalogue-group').firstElementChild.outerHTML+options;
  renderCatalogue();
  return c;
});
catalogueReady.catch(e=>notice(e.message,true));
$('catalogue-group').onchange=renderCatalogue;
$('catalogue-review').onclick=handler(()=>changeTab('intervals'));
$('review-catalogue').onclick=handler(e=>{e.preventDefault();return changeTab('catalogue');});
const caseBase=()=>'/api/cases/'+encodeURIComponent(intervalState.current.id);
function resetCase(){
  intervalState.current=null;intervalState.reviewRequest++;
  $('case-content').hidden=true;$('case-assessment').hidden=true;
  $('case-status').textContent='';
}
async function refreshIntervalCases(preferred){
  const request=++intervalState.request;
  resetCase();$('case-create-button').disabled=true;
  $('case-select').innerHTML='<option value="">No interval selected</option>';
  await catalogueReady;
  if(request!==intervalState.request||!state.pid)return;
  const query=new URLSearchParams({dataset_id:state.manifest.id,player_id:state.pid});
  const cases=await api('/api/cases?'+query);
  if(request!==intervalState.request)return;
  $('case-create-button').disabled=false;
  $('case-period').value=state.manifest.half||1;
  $('case-start').value=state.manifest.video_start_s||0;
  $('case-end').value=Math.min(Number($('case-start').value)+1200,state.manifest.duration_seconds);
  $('case-select').innerHTML='<option value="">Choose an interval</option>'+cases.map(c=>`<option value="${esc(c.id)}">Period ${c.period} · ${c.start_s}–${c.end_s}s · ${esc(c.position_group.replaceAll('_',' '))}${c.stale?' · source changed':''}</option>`).join('');
  $('case-status').textContent=cases.length?`${cases.length} saved intervals for ${state.player.name}.`:'No reviewed intervals for this player yet.';
  if(preferred){$('case-select').value=preferred;await selectIntervalCase(preferred);}
}
function addSequence(sequence={}){
  const row=document.createElement('div');row.className='form-grid sequence-row';
  row.innerHTML=`<label>Sequence start (s)<input class="seq-start" type="number" min="0" step="0.1" value="${esc(sequence.start_s??'')}"></label><label>Sequence end (s)<input class="seq-end" type="number" min="0" step="0.1" value="${esc(sequence.end_s??'')}"></label><label>Observed behaviour<input class="seq-note" maxlength="2000" value="${esc(sequence.note??'')}"></label><button type="button">Remove sequence</button>`;
  row.querySelector('button').onclick=()=>row.remove();$('case-sequences').append(row);
}
function updateMixturePreview(){
  if(!intervalState.current){$('case-mixture').innerHTML='';return;}
  const rated=intervalState.current.labels.map(id=>({id,label:names[id]||id,
    value:$('case-unknown-'+id).checked?null:Number($('case-label-'+id).value)})).filter(e=>e.value>0);
  const total=rated.reduce((sum,e)=>sum+e.value,0)||1;
  $('case-mixture').innerHTML=bars(rated.sort((a,b)=>b.value-a.value).map(e=>({label:e.label,value:e.value/total*100})));
}
function resetIntervalReview(){
  for(const cb of $('case-labels').querySelectorAll('.rating-unknown'))cb.checked=true;
  for(const slider of $('case-labels').querySelectorAll('input[type=range]')){slider.value=0;slider.disabled=true;}
  $('case-labels').querySelectorAll('output').forEach(o=>o.textContent='—');
  $('case-confidence').value='uncertain';
  for(const id of ['case-evidence','case-abstain','case-notes'])$(id).value='';
  $('case-sequences').replaceChildren();for(let i=0;i<3;i++)addSequence();
  updateMixturePreview();
}
async function loadIntervalReview(){
  const request=++intervalState.reviewRequest;
  resetIntervalReview();$('case-assessment').hidden=true;
  const reviewer=$('case-reviewer').value.trim(),id=intervalState.current?.id;
  if(!reviewer||!id)return;
  const result=await api(caseBase()+'/reviews?reviewer='+encodeURIComponent(reviewer));
  if(request!==intervalState.reviewRequest||id!==intervalState.current?.id)return;
  const review=result.own_review;if(!review)return;
  for(const [key,value] of Object.entries(review.labels)){
    const cb=$('case-unknown-'+key),slider=$('case-label-'+key),out=$('case-value-'+key);
    if(!cb)continue;
    cb.checked=value==null;slider.disabled=value==null;slider.value=value==null?0:value;out.textContent=value==null?'—':value+'%';
  }
  $('case-confidence').value=review.confidence;
  $('case-evidence').value=review.evidence;$('case-abstain').value=review.abstain_reason;$('case-notes').value=review.notes;
  $('case-sequences').replaceChildren();review.sequences.forEach(addSequence);
  updateMixturePreview();
}
async function selectIntervalCase(id){
  const request=++intervalState.request;resetCase();if(!id)return;
  const c=await api('/api/cases/'+encodeURIComponent(id));
  if(request!==intervalState.request)return;
  intervalState.current=c;
  const roles=intervalState.catalogue.roles.filter(r=>c.labels.includes(r.id));
  $('case-labels').innerHTML=roles.map(r=>`<label class="rating-row">${esc(r.name)}${r.trainable?'':' (descriptive only)'}<small>${esc(r.definition)}</small>${roleExamples(r)}<small>${esc(r.evidence_cues)}</small><div class="rating-control"><input type="checkbox" class="rating-unknown" id="case-unknown-${esc(r.id)}" checked><span class="small muted">Insufficient evidence</span><input type="range" min="0" max="100" step="5" value="0" id="case-label-${esc(r.id)}" disabled><output id="case-value-${esc(r.id)}">—</output></div></label>`).join('');
  $('case-labels').querySelectorAll('.rating-unknown').forEach(cb=>{
    const roleId=cb.id.slice('case-unknown-'.length),slider=$('case-label-'+roleId),out=$('case-value-'+roleId);
    cb.onchange=()=>{slider.disabled=cb.checked;out.textContent=cb.checked?'—':slider.value+'%';updateMixturePreview();};
    slider.oninput=()=>{out.textContent=slider.value+'%';updateMixturePreview();};
  });
  const options=roles.map(r=>`<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');
  $('case-adj-label').innerHTML=options;
  $('case-content').hidden=false;
  $('case-status').textContent=c.stale?'Source data changed. Create a new case and review the revised evidence.':`${c.profile.name} · period ${c.period} · ${c.start_s}–${c.end_s}s. ${c.pilot_duration_met?'20-minute duration requirement met.':'Shorter than the 20-minute pilot minimum; excluded from training.'}`;
  pitch($('case-heatmap'),c.profile.heatmap);
  $('case-quality').textContent=`${pct(c.profile.position_coverage)} observed position coverage · ${num(c.profile.observed_seconds/60)} observed minutes · ${c.profile.direction_known?'attack direction confirmed':'attack direction unconfirmed'}. ${c.profile.feature_events_note}`;
  $('case-media').innerHTML=mediaMarkup(false);
  const video=$('case-media').querySelector('video');
  if(video){const offset=(state.manifest.source_offset_s||0)-(state.manifest.video_start_s||0);video.addEventListener('loadedmetadata',()=>{video.currentTime=c.start_s+offset;},{once:true});video.addEventListener('timeupdate',()=>{if(!video.paused&&video.currentTime>=c.end_s+offset)video.pause();});}
  await loadIntervalReview();
  if(request!==intervalState.request)return;
  $('case-review').querySelectorAll('input,select,textarea,button').forEach(e=>e.disabled=c.stale);
  $('case-adjudicate').querySelectorAll('input,select,textarea,button').forEach(e=>e.disabled=c.stale);
}
function showIntervalAssessment(result){
  $('case-assessment').hidden=false;
  const c=result.consensus;
  const rows=Object.entries(c.labels).map(([key,v])=>({label:names[key]||key,value:v.value,
    sub:v.status+(v.spread!=null?` · spread ${num(v.spread)}pt`:'')+(v.reviewer_count?` · ${v.reviewer_count} reviewer${v.reviewer_count===1?'':'s'}`:'')}));
  let html='<h3>Reviewer consensus</h3>'+bars(rows.sort((a,b)=>(b.value??-1)-(a.value??-1)));
  html+=`<p class="small muted">${c.primary_role?'Primary: '+esc(names[c.primary_role]||c.primary_role):'No settled primary role yet'}${c.secondary_traits.length?'. Secondary: '+c.secondary_traits.map(k=>esc(names[k]||k)).join(', '):''}</p>`;
  if(result.prediction)html+='<h3>Model suggestion</h3>'+predictionMarkup(result.prediction);
  $('case-assessment').innerHTML=html;
}
$('case-reviewer').value=localStorage.getItem('pitchprofile-reviewer')||'';
$('case-reviewer').onchange=handler(async()=>{localStorage.setItem('pitchprofile-reviewer',$('case-reviewer').value);await loadIntervalReview();});
$('case-select').onchange=handler(e=>selectIntervalCase(e.target.value));
$('case-create').onsubmit=handler(async e=>{
  e.preventDefault();if(!state.pid)return;
  const dataset_id=state.manifest.id,player_id=state.pid;
  const c=await post('/api/cases',{dataset_id,player_id,period:Number($('case-period').value),start_s:Number($('case-start').value),end_s:Number($('case-end').value),position_group:$('case-group').value,context:$('case-context').value});
  if(dataset_id===state.manifest?.id&&player_id===state.pid)await refreshIntervalCases(c.id);
  notice('Interval case saved. Review its matching evidence independently.');
});
$('case-add-sequence').onclick=()=>addSequence();
$('case-review').onsubmit=handler(async e=>{
  e.preventDefault();const labels={};
  for(const key of intervalState.current.labels){
    labels[key]=$('case-unknown-'+key).checked?null:Number($('case-label-'+key).value);
  }
  const sequences=[...$('case-sequences').children].filter(row=>[...row.querySelectorAll('input')].some(i=>i.value!=='')).map(row=>{
    if([...row.querySelectorAll('input')].some(i=>i.value.trim()===''))throw Error('Complete the start, end and note for each sequence, or remove its row.');
    return {start_s:Number(row.querySelector('.seq-start').value),end_s:Number(row.querySelector('.seq-end').value),note:row.querySelector('.seq-note').value};
  });
  await post(caseBase()+'/review',{reviewer:$('case-reviewer').value,labels,sequences,confidence:$('case-confidence').value,evidence:$('case-evidence').value,abstain_reason:$('case-abstain').value,notes:$('case-notes').value});
  $('case-assessment').hidden=true;notice('Independent interval review saved.');
});
$('case-assess').onclick=handler(async()=>showIntervalAssessment(await api(caseBase()+'/assessment')));
$('case-adjudicate').onsubmit=handler(async e=>{e.preventDefault();const consensus=await post(caseBase()+'/adjudicate',{label:$('case-adj-label').value,value:Number($('case-adj-value').value),reviewer:$('case-adj-reviewer').value,reason:$('case-adj-reason').value});showIntervalAssessment({consensus});notice('Interval adjudication saved.');});
window.addEventListener('profilechange',handler(()=>refreshIntervalCases()));
window.addEventListener('profilechanging',()=>{
  intervalState.request++;resetCase();$('case-create-button').disabled=true;
  $('case-select').innerHTML='<option value="">Loading player…</option>';
});
refreshIntervalCases().catch(e=>notice(e.message,true));
