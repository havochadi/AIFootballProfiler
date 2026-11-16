'use strict';

// Group C's read-only contribution to the Live model view. Identity evidence belongs to the
// selected player track as a whole; it is not a new inference at the current video time.
const IDENTITY_LIVE_STATUS = {
  confirmed: p => `Identity confirmed by ${p.identity_reviewer || 'a reviewer'}`,
  corrected: p => `Identity corrected by ${p.identity_reviewer || 'a reviewer'}`,
  lineup_match: () => 'Automatic shirt number matched one player in the line-up · not visually confirmed',
  unused_substitute: () => 'Automatic shirt number matched an unused substitute · check this identity',
  not_in_lineup: () => 'Automatic shirt number was not found in the match-day squad',
  kit_unmapped: () => 'Automatic identity · kit group is not linked to a team',
  no_context: () => 'Automatic identity · no match context is connected',
};

function identityLiveValue(value, fallback='—') {
  return value == null || value === '' ? fallback : String(value);
}

function identityLiveRow(ctx) {
  const p=ctx?.player?.(ctx.pid);
  if(!p)return '<p class="small muted">No selected player identity is available.</p>';

  const team=p.team_name||ctx.teamName?.(p.team)||p.team||'Unknown team';
  const jersey=p.jersey==null?'':` · #${esc(identityLiveValue(p.jersey))}`;
  const status=IDENTITY_LIVE_STATUS[p.identity_status]?.(p)||'Automatic identity';
  const evidence=[];
  if(p.automatic_jersey!=null){
    const automatic=`Automatic reader: #${identityLiveValue(p.automatic_jersey)}`;
    evidence.push(p.jersey!=null&&String(p.jersey)!==String(p.automatic_jersey)
      ?`${automatic} · current identity: #${identityLiveValue(p.jersey)}`:automatic);
  }

  const share=p.number_guess_share??p.guess_share;
  const hasReadableViews=p.readable_views!=null&&p.readable_views!=='';
  const readableViews=Number(p.readable_views);
  if(hasReadableViews&&p.number_guess!=null&&Number.isFinite(readableViews)&&readableViews>0){
    const numericShare=Number(share);
    const validShare=share!=null&&share!==''&&Number.isFinite(numericShare)&&numericShare>=0&&numericShare<=1;
    const votes=validShare?` · ${Math.round(numericShare*100)}% of readable-view votes`:'';
    evidence.push(`Reader's best guess: #${identityLiveValue(p.number_guess)} · ${identityLiveValue(p.readable_views)} readable views${votes}`);
  }else if(hasReadableViews&&Number.isFinite(readableViews)&&readableViews>0){
    evidence.push(`No clear shirt-number guess · ${identityLiveValue(p.readable_views)} readable views`);
  }else if(p.readable_views===0){
    evidence.push('No readable shirt-number views were found');
  }

  return `<div class="lm-who"><strong>${esc(identityLiveValue(p.name))}</strong>${jersey} <span class="muted">${esc(identityLiveValue(team))}</span></div>
    <div class="small muted">${esc(status)}</div>${evidence.map(line=>`<div class="small muted">${esc(line)}</div>`).join('')}`;
}

LiveModels.addRow({id:'identity',title:'Who is this · Group C',order:30,render:identityLiveRow});
