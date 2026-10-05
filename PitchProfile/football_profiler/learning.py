from __future__ import annotations
import copy
import hashlib
import itertools
import json
import time
import warnings
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit
from . import storage as S
from .features import case_profiles, drop_observations, profile, FEATURE_NAMES, GRID
from .runtime import torch_device

def agreement(schema_version="1.0"):
    from . import cases as C, taxonomy as T
    modern=schema_version==T.VERSION
    label_names=T.LABELS if modern else S.LABELS
    grouped={}
    source=C.reviews() if modern else S.reviews()
    for r in source:grouped.setdefault(r.get('case_id',(r.get("dataset_id"),r.get("player_id"))),[]).append(r)
    result=[]
    for name in label_names:
        pairs={}
        for rows in grouped.values():
            for a,b in itertools.combinations(sorted(rows,key=lambda x:x["reviewer"]),2):
                x,y=a["labels"].get(name),b["labels"].get(name)
                if x is not None and y is not None:pairs.setdefault((a["reviewer"],b["reviewer"]),[]).append((x,y))
        for reviewers,values in pairs.items():
            a,b=np.asarray(values,dtype=float).T
            if modern:
                # Percentage ratings are continuous: Cohen's kappa and exact-match agreement don't
                # apply. Reuse the same tolerance band as consensus(), plus a correlation in its place.
                agree=float(np.mean(np.abs(a-b)<=T.AGREEMENT_TOLERANCE))
                k=float(np.corrcoef(a,b)[0,1]) if len(values)>1 and a.std()>0 and b.std()>0 else None
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kappa=cohen_kappa_score(a,b,labels=[0,1]) if len(values)>1 else np.nan
                agree,k=float(np.mean(a==b)),float(kappa) if np.isfinite(kappa) else None
            result.append({"label":name,"reviewers":reviewers,"cases":len(values),"raw_agreement":agree,"kappa":k})
    note=(f"Independent percentage ratings before adjudication. raw_agreement is the share of paired "
          f"ratings within {T.AGREEMENT_TOLERANCE} points of each other; kappa here is a Pearson "
          "correlation between the two reviewers' ratings (undefined when either rated everything "
          "identically), not literally Cohen's kappa. Inspect case counts before drawing conclusions."
         ) if modern else "Independent ratings before adjudication. Undefined kappa remains unavailable; inspect prevalence and case counts."
    return {"rubric_version":schema_version,"pairwise_results":result,"note":note}

def split_groups(groups,seed=42):
    groups=np.asarray(groups)
    if len(set(groups))<3:raise ValueError("At least three independent groups are needed for training, validation and test")
    index=np.arange(len(groups))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(index,groups=groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(trainval,groups=groups[trainval]))
    train,val=trainval[tr],trainval[va]
    return train,val,test

def metrics(y,scores,thresholds,active,label_names=S.LABELS):
    rows=[];f=[]
    for j,name in enumerate(label_names):
        mask=np.isfinite(y[:,j])
        # y is a continuous [0,1] percentage rating; threshold at 50% for classification-style
        # precision/recall/F1, but also report mean absolute error on the untouched scale.
        target=(y[mask,j]>=.5).astype(int)
        entry={"label":name,"known_cases":int(mask.sum()),"positive_cases":int(target.sum()),"threshold":float(thresholds[j]),"trained":bool(active[j])}
        if active[j] and mask.any():
            pred=(scores[mask,j]>=thresholds[j]).astype(int)
            p,r,ff,_=precision_recall_fscore_support(target,pred,average="binary",zero_division=0)
            mae=float(np.mean(np.abs(scores[mask,j]-y[mask,j])))*100
            entry.update(precision=float(p),recall=float(r),f1=float(ff),both_classes_in_test=len(set(target))==2,mean_abs_error_points=mae)
            if len(set(target))==2:f.append(float(ff))
        rows.append(entry)
    return {"macro_f1":float(np.mean(f)) if f else None,"labels_with_both_classes":len(f),"per_label":rows,"note":"Macro F1 averages supported labels with both test classes (thresholded at 50%). mean_abs_error_points measures the continuous percentage rating directly, in points out of 100. Unknown labels are excluded; this is not external expert validation."}

def case_splits(cases, groups, seed, split_by):
    declared = [p.get('benchmark_split') for p in cases]
    if not any(declared):
        return split_groups(groups, seed)
    if split_by != 'match' or any(s not in ('train', 'validation', 'test') for s in declared):
        raise ValueError('Provider benchmark data requires a separate match experiment with declared train, validation and test partitions for every case')
    partitions = [np.array([i for i, s in enumerate(declared) if s == name], dtype=int)
                  for name in ('train', 'validation', 'test')]
    if any(not len(indices) for indices in partitions):
        raise ValueError('Review cases in all three provider benchmark partitions before evaluation; held-out matches cannot be reassigned')
    memberships = {}
    for group, partition in zip(groups, declared):
        memberships.setdefault(group, set()).add(partition)
    if any(len(s) > 1 for s in memberships.values()):
        raise ValueError('The same match occurs in multiple provider benchmark partitions')
    return tuple(partitions)

def build_network(output_labels=None):
    import torch.nn as nn
    class ProfileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten())
            self.head=nn.Sequential(nn.Linear(256+len(FEATURE_NAMES),32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,output_labels or len(S.LABELS)))
        def forward(self,h,x):
            import torch
            return self.head(torch.cat([self.encoder(h),x],dim=1))
    return ProfileNet()

def masked_loss(logits,y,pos_weight=None):
    import torch
    mask=torch.isfinite(y);target=torch.nan_to_num(y,nan=0.)
    losses=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none",pos_weight=pos_weight)
    return (losses*mask).sum()/mask.sum().clamp_min(1)

def labelled_cases(schema_version="1.0"):
    from . import cases as C, taxonomy as T
    if schema_version==T.VERSION:return C.training_profiles()
    if schema_version!="1.0":raise ValueError("Unknown rubric version")
    cases=[]
    for p in case_profiles():
        c=S.consensus(p["dataset_id"],p["player_id"])
        values=[c["labels"][name]["value"] for name in S.LABELS]
        if (p["features_available"] and p["direction_known"]
                and (p["position_coverage"] or 0)>=.2 and p["observed_seconds"]>=30
                and any(x is not None for x in values)):
            cases.append({**p,"y":[np.nan if x is None else x for x in values]})
    return cases

def train_models(split_by="match",epochs=40,seed=42,progress=lambda *a:None,schema_version="1.0"):
    from . import cases as C, taxonomy as T
    label_names=T.LABELS if schema_version==T.VERSION else S.LABELS
    nlabels=len(label_names)
    import torch
    torch.set_num_threads(2)
    cases=labelled_cases(schema_version)
    if len(cases)<12:raise ValueError(f"Only {len(cases)} usable reviewed cases. Add at least 12 cases with independent agreement, calibrated positions and confirmed attacking direction. These are minimum execution checks, not a claim of sufficient data.")
    if split_by not in ("match","player"):raise ValueError("Split by match or player")
    if split_by=="player" and any(not p.get("identity_verified") or not p.get("global_id") for p in cases):raise ValueError("Player holdout requires verified persistent identities for every case")
    groups=[p["match_id"] if split_by=="match" else p["global_id"] for p in cases]
    tr,va,te=case_splits(cases,groups,seed,split_by)
    y=np.asarray([p["y"] for p in cases],np.float32)
    # Targets are continuous in [0,1] (percentage ratings) as of taxonomy v2; a 50% cut is
    # used only to judge whether a label has contrasting support, never to train on.
    active=np.array([np.sum(y[tr,j]>=.5)>=2 and np.sum(y[tr,j]<.5)>=2 for j in range(nlabels)])
    if not active.any():raise ValueError("The training split needs at least two positive and two negative examples for a label. Expand the labelled cohort.")
    device=torch_device()
    # Inactive heads cannot silently learn from unknown or one-class labels.
    y[:,~active]=np.nan
    x=np.array([p["feature_vector"] for p in cases],np.float32)
    mu=x[tr].mean(0);sd=x[tr].std(0);sd[sd<1e-6]=1
    base_h=np.asarray([p["heatmap"] for p in cases],np.float32)[:,None]
    variants={0.:(base_h,x)}
    raw={d:S.load_tracks(d) for d in {p["dataset_id"] for p in cases}}
    manifests={d:S.read_json(S.dataset_dir(d)/"manifest.json") for d in raw}
    for gap in [.2,.4,.6]:
        hh=[];xx=[]
        for i,p in enumerate(cases):
            rows=raw[p["dataset_id"]];rows=rows[rows.player_id.eq(p["player_id"])]
            if schema_version==T.VERSION:rows=C.interval_rows(rows,p)
            masked=drop_observations(rows,gap,np.random.default_rng(seed+1000+i))
            z=profile(masked,p,manifests[p["dataset_id"]],p.get("events"))
            hh.append(z["heatmap"]);xx.append(z["feature_vector"])
        variants[gap]=(np.asarray(hh,np.float32)[:,None],np.asarray(xx,np.float32))
    run_id=datetime.now(timezone.utc).strftime("archetypes-%Y%m%d-%H%M%S-%f")
    out=S.DATA/"models"/run_id;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"scaler.npz",mean=mu,scale=sd)
    fit_targets=torch.tensor(y,device=device)
    tensor_variants={gap:(torch.as_tensor(hh,device=device),torch.as_tensor((xx-mu)/sd,device=device)) for gap,(hh,xx) in variants.items()}
    validation_index=torch.as_tensor(va,dtype=torch.long,device=device)
    positive=np.sum(y[tr]>=.5,axis=0);negative=np.sum(y[tr]<.5,axis=0)
    weights=torch.tensor(np.clip(negative/np.maximum(positive,1),.1,10),dtype=torch.float32,device=device)
    predictions={};histories={}
    coef=np.zeros((nlabels,len(FEATURE_NAMES)));intercept=np.zeros(nlabels)
    for j in range(nlabels):
        if not active[j]:continue
        known=tr[np.isfinite(y[tr,j])]
        model=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed)
        # Logistic regression is a binary baseline; the CNN below trains on the true
        # continuous percentage target via soft-label BCE (masked_loss accepts [0,1] targets).
        model.fit((x[known]-mu)/sd,(y[known,j]>=.5).astype(int))
        coef[j]=model.coef_[0];intercept[j]=model.intercept_[0]
    np.savez_compressed(out/"logistic.npz",coef=coef,intercept=intercept)
    for gap,(_,xx) in variants.items():
        logits=np.clip(((xx-mu)/sd)@coef.T+intercept,-30,30)
        predictions.setdefault("logistic",{})[gap]=1/(1+np.exp(-logits))
    if device.type=="cuda":torch.cuda.synchronize(device)
    start=time.perf_counter()
    for mode_index,mode in enumerate(["cnn","cnn_with_gaps"]):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=(build_network(nlabels) if schema_version==T.VERSION else build_network()).to(device)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        best=copy.deepcopy(net.state_dict());best_loss=float("inf");stale=0;history=[]
        for epoch in range(epochs):
            net.train();losses=[]
            for indices in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/16)))):
                gap=float(rng.choice([0,.2,.4,.6])) if mode=="cnn_with_gaps" else 0.
                hh,xx=tensor_variants[gap]
                batch_index=torch.as_tensor(indices,dtype=torch.long,device=device)
                logits=net(hh[batch_index],xx[batch_index])
                loss=masked_loss(logits,fit_targets[batch_index],weights)
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
            net.eval()
            hh,xx=tensor_variants[0.]
            with torch.no_grad():vl=float(masked_loss(net(hh[validation_index],xx[validation_index]),fit_targets[validation_index],weights))
            history.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation_loss":vl})
            if vl<best_loss-1e-5:best_loss=vl;best=copy.deepcopy(net.state_dict());stale=0
            else:stale+=1
            progress(.2+.7*(mode_index+((epoch+1)/epochs))/2,f"Training {mode}: epoch {epoch+1}")
            if stale>=8:break
        net.load_state_dict(best);net.eval()
        # CPU tensors make checkpoints portable across CUDA devices and explicit CPU runs.
        torch.save({key:value.detach().cpu() for key,value in net.state_dict().items()},out/(mode+".pt"));histories[mode]=history
        predictions[mode]={}
        with torch.no_grad():
            for gap,(hh,xx) in tensor_variants.items():predictions[mode][gap]=net(hh,xx).sigmoid().cpu().numpy()
    if device.type=="cuda":torch.cuda.synchronize(device)
    training_seconds=time.perf_counter()-start
    results={};thresholds={}
    # Select thresholds exclusively on the untouched validation split.
    for name,by_gap in predictions.items():
        th=np.full(nlabels,.5)
        for j in range(nlabels):
            valid=va[np.isfinite(y[va,j])]
            if active[j] and len(set(y[valid,j]))==2:
                candidates=np.linspace(.2,.8,13)
                def score(t):return precision_recall_fscore_support(y[valid,j],by_gap[0.][valid,j]>=t,average="binary",zero_division=0)[2]
                th[j]=float(max(candidates,key=score))
        thresholds[name]=th
        results[name]={str(g):metrics(y[te],scores[te],th,active,label_names) for g,scores in by_gap.items()}
    val_scores={name:metrics(y[va],p[0.][va],thresholds[name],active,label_names)["macro_f1"] for name,p in predictions.items()}
    selected=max(val_scores,key=lambda n:val_scores[n] if val_scores[n] is not None else -1)
    info={"run_id":run_id,"task":"Team-defined archetype prediction","created":S.now(),"seed":seed,"split_by":split_by,"cases":len(cases),"feature_names":FEATURE_NAMES,"active_labels":active,"thresholds":thresholds,"selected_model":selected,"validation_macro_f1":val_scores,"results":results,"splits":{name:[{"case_id":cases[i]["case_id"],"group":groups[i]} for i in indices] for name,indices in [("train",tr),("validation",va),("test",te)]},"label_snapshot":S.reviews(),"label_digest":hashlib.sha256(json.dumps(S.reviews(),sort_keys=True).encode()).hexdigest(),"histories":histories,"training_seconds":training_seconds,"device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU","baseline_device":"cpu","limitations":["Team reference labels are not external expert ground truth.","Holdout by match and by player are different experiments.","Model scores are not calibrated probabilities.","Gap tests remove supplied observations, not true off-screen trajectories."]}
    snapshot=C.reviews() if schema_version==T.VERSION else S.reviews()
    info.update(rubric_version=schema_version,label_names=label_names,label_snapshot=snapshot,
                label_digest=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest())
    if schema_version==T.VERSION:
        info['catalogue']=T.CATALOGUE
        info['case_snapshot']=[{k:p[k] for k in ('case_id','dataset_id','player_id','period','start_s','end_s','position_group')} for p in cases]
        info['cohort_support']=C.summary()['support']
        info['limitations'].append('Pilot execution thresholds do not establish sufficient support for each of the 42 catalogue roles.')
    S.write_json(out/"report.json",info);S.write_json(S.DATA/("models/latest-v2.json" if schema_version==T.VERSION else "models/latest.json"),{"run_id":run_id})
    progress(1,"Training and held-out evaluation complete")
    return S.clean_json(info)

def predict(p):
    from . import taxonomy as T
    modern=p.get('taxonomy_version')==T.VERSION
    # 'reviewed_seconds' only exists on interval-case profiles (the 20-minute pilot rule);
    # whole-player profiles never set it and must fall through to the observed_seconds gate below.
    if modern and 'reviewed_seconds' in p and p['reviewed_seconds']<T.CATALOGUE['pilot']['reviewed_seconds']:
        return {"status":"insufficient_evidence","message":"The proposal pilot requires a reviewed interval of at least 20 minutes."}
    latest=S.read_json(S.DATA/("models/latest-v2.json" if modern else "models/latest.json"))
    if not latest:return {"status":"not_trained","message":"Team reference labels are required before archetype predictions can be trained."}
    if not p["features_available"] or not p["direction_known"] or (p["position_coverage"] or 0)<.2 or p["observed_seconds"]<30:
        return {"status":"insufficient_evidence","message":"Confirm attacking direction and obtain at least 30 observed seconds with 20% position coverage. These are provisional quality thresholds."}
    out=S.DATA/"models"/latest["run_id"];info=S.read_json(out/"report.json");kind=info["selected_model"]
    if info["validation_macro_f1"][kind] is None:return {"status":"insufficient_validation","message":"The validation split does not contain both classes for any supported label. Expand the reviewed cohort."}
    scaler=np.load(out/"scaler.npz");x=(np.array(p["feature_vector"],np.float32)-scaler["mean"])/scaler["scale"]
    if kind=="logistic":
        state=np.load(out/"logistic.npz");z=np.clip(state["coef"]@x+state["intercept"],-30,30);scores=1/(1+np.exp(-z))
    else:
        import torch
        device=torch_device()
        net=(build_network(len(info['label_names'])) if modern else build_network()).to(device);net.load_state_dict(torch.load(out/(kind+".pt"),map_location=device,weights_only=True));net.eval()
        with torch.no_grad():scores=net(torch.tensor(np.array(p["heatmap"],np.float32)[None,None],device=device),torch.tensor(x[None],device=device)).sigmoid().cpu().numpy()[0]
    label_names=info.get('label_names',S.LABELS)
    allowed=set(T.compatible(p['position_group'])) if modern else set(label_names)
    supported_total=sum(float(scores[j]) for j,name in enumerate(label_names) if name in allowed and info["active_labels"][j]) or 1.
    return {"status":"model_suggestion","run_id":info["run_id"],"model":kind,
            "labels":[{"label":name,
                       "percentage":float(scores[j])*100 if info["active_labels"][j] else None,
                       "share":float(scores[j])/supported_total if info["active_labels"][j] else None,
                       "threshold":info["thresholds"][kind][j]*100,
                       "suggested":bool(scores[j]>=info["thresholds"][kind][j]) if info["active_labels"][j] else None}
                      for j,name in enumerate(label_names) if name in allowed],
            "note":"Model scores measure learned agreement with team labels, shown as a percentage; they are not calibrated real-world probabilities. 'share' renormalises the same scores across supported roles so they read as a mixture summing to 100%."}
