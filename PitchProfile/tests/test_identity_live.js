'use strict';

// Standalone: node PitchProfile/tests/test_identity_live.js
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

const rows=[];
const esc=value=>String(value).replace(/[&<>"']/g,c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
})[c]);
const source=fs.readFileSync(path.join(__dirname,'../static/identity_live.js'),'utf8');
vm.runInNewContext(source,{LiveModels:{addRow:row=>rows.push(row)},esc});

assert.equal(rows.length,1);
const row=rows[0];
assert.equal(row.id,'identity');
assert.equal(row.order,30);
assert.equal(row.title,'Who is this · Group C');

const context=(player)=>({pid:'A-X1',player:()=>player,teamName:()=>'<Team & One>'});
assert.match(row.render(context(null)),/No selected player identity is available/);

const corrected=row.render(context({
  name:'Alex <script>alert(1)</script>',team:'A',jersey:35,automatic_jersey:99,
  identity_status:'corrected',identity_reviewer:'R & D',number_guess:35,
  number_guess_share:.625,readable_views:8,
}));
assert.match(corrected,/Alex &lt;script&gt;alert\(1\)&lt;\/script&gt;/);
assert.doesNotMatch(corrected,/<script>/);
assert.match(corrected,/&lt;Team &amp; One&gt;/);
assert.match(corrected,/Identity corrected by R &amp; D/);
assert.match(corrected,/Automatic reader: #99 · current identity: #35/);
assert.match(corrected,/63% of readable-view votes/);

const lineup=row.render(context({
  name:'Player Seven',team:'A',jersey:7,automatic_jersey:7,
  identity_status:'lineup_match',number_guess:7,guess_share:.4,readable_views:5,
}));
assert.match(lineup,/Automatic shirt number matched one player in the line-up/);
assert.match(lineup,/not visually confirmed/);
assert.match(lineup,/40% of readable-view votes/);
assert.doesNotMatch(lineup,/accuracy|confidence/i);

const noGuess=row.render(context({
  name:'Unnamed player',team:'A',identity_status:'no_context',readable_views:6,
}));
assert.match(noGuess,/No clear shirt-number guess · 6 readable views/);

for(const invalidShare of [-.1,1.1,'not-a-number','']){
  const invalid=row.render(context({
    name:'Player Eight',team:'A',number_guess:8,number_guess_share:invalidShare,readable_views:4,
  }));
  assert.match(invalid,/Reader&#39;s best guess: #8 · 4 readable views/);
  assert.doesNotMatch(invalid,/% of readable-view votes/);
}

const zeroViews=row.render(context({
  name:'Unread player',team:'A',number_guess:4,number_guess_share:.8,readable_views:0,
}));
assert.match(zeroViews,/No readable shirt-number views were found/);
assert.doesNotMatch(zeroViews,/Reader&#39;s best guess/);

const nullViews=row.render(context({
  name:'Unknown evidence',team:'A',number_guess:4,number_guess_share:.8,readable_views:null,
}));
assert.doesNotMatch(nullViews,/No readable shirt-number views were found/);
assert.doesNotMatch(nullViews,/readable views/);

const confirmed=row.render(context({
  name:'Confirmed Player',team:'A',jersey:10,identity_status:'confirmed',identity_reviewer:'Human Reviewer',
}));
assert.match(confirmed,/Identity confirmed by Human Reviewer/);
assert.doesNotMatch(confirmed,/Automatic identity/);

console.log('identity_live renderer: all checks passed');
