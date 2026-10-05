"""Observed motion and visible-team geometry, never inferred off-screen movement."""
from __future__ import annotations
import io
import zipfile
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from . import storage as S
from .features import observed_points

ZONES = [('walk_or_stationary', 0, 7.2), ('jog', 7.2, 14.4),
         ('run', 14.4, 19.8), ('high_speed', 19.8, float('inf'))]


def motion(rows, sampling_hz):
    q = observed_points(rows).sort_values('time_s')
    xy = q[['x','y']].to_numpy()
    times = q.time_s.to_numpy()
    dt = np.diff(times)
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speeds = distances / np.maximum(dt, 1e-9)
    good = (dt > 0) & (dt <= 1.5 / sampling_hz + 1e-6) & (speeds <= 12)
    if 'period' in q:
        good &= np.diff(q.period.to_numpy()) == 0
    if 'track_id' in q:
        good &= q.track_id.to_numpy()[1:] == q.track_id.to_numpy()[:-1]
    elapsed = float(dt[good].sum())
    distance = float(distances[good].sum())
    middle = (times[:-1] + times[1:]) / 2
    accel_good = good[1:] & good[:-1] & (np.diff(middle) > 0)
    acceleration = np.diff(speeds) / np.maximum(np.diff(middle), 1e-9)
    trace = [{'time_s': float(t), 'speed_kmh': float(s*3.6) if ok else None}
             for t,s,ok in zip(times[1:], speeds, good)]
    stride = max(1, int(np.ceil(len(trace)/1200)))
    return {'valid_steps': int(good.sum()), 'excluded_steps': int((~good).sum()),
            'observed_motion_seconds': elapsed, 'distance_m': distance if elapsed else None,
            'mean_speed_kmh': distance / elapsed * 3.6 if elapsed else None,
            'peak_speed_kmh': float(speeds[good].max()*3.6) if elapsed else None,
            'peak_abs_acceleration_ms2': float(np.abs(acceleration[accel_good]).max()) if accel_good.any() else None,
            'zones': [{'id': name, 'lower_kmh': lo, 'upper_kmh': hi if np.isfinite(hi) else None,
                       'seconds': float(dt[good & (speeds*3.6 >= lo) & (speeds*3.6 < hi)].sum())}
                      for name,lo,hi in ZONES], 'speed_trace': trace[::stride],
            'note': 'Observed segments only; missing frames, track/period changes and speeds above 12 m/s are excluded. No full-match extrapolation. Raw coordinate noise affects speed and acceleration. Intensity thresholds are configurable heuristics.'}


def stadium_points(rows, manifest):
    q = observed_points(rows).copy()
    kind = manifest.get('source_kind')
    if manifest.get('coordinate_orientation') == 'stadium' or kind == 'reference_annotations':
        return q
    by_id = {str(p['player_id']): p for p in manifest['players']}
    reversible = kind in ('model_predictions', 'soccertrack_reference')
    if not reversible and any('stadium_direction' not in by_id[pid] for pid in q.player_id.unique()):
        return q.iloc[:0]
    for pid in q.player_id.unique():
        person = by_id[pid]
        direction = person.get('direction') if reversible else person.get('stadium_direction')
        if direction == 'left':
            chosen = q.player_id.eq(pid)
            q.loc[chosen, ['x','y']] = np.array([105,68]) - q.loc[chosen, ['x','y']]
    return q


def snapshot(rows, manifest, seconds):
    q = stadium_points(rows, manifest)
    if q.empty:
        return {'time_s': None, 'players': [], 'teams': [],
                'note': 'Common pitch orientation or calibrated observed positions are unavailable for this dataset.'}
    closest = q.loc[(q.time_s-seconds).abs().idxmin()]
    if abs(float(closest.time_s)-seconds) > 1.5/manifest['sampling_hz']:
        return {'time_s': None, 'players': [], 'teams': [], 'note': 'No observed positions near this timestamp.'}
    q = q[q.frame.eq(closest.frame)]
    people = {str(p['player_id']): p for p in manifest['players']}
    points, teams = [], []
    for row in q.itertuples():
        person = people[row.player_id]
        points.append({'player_id': row.player_id, 'name': person.get('name',row.player_id),
                       'team': person.get('team','Unconfirmed'), 'role': person.get('role','Unconfirmed'),
                       'position_group': person.get('position_group'), 'x': row.x, 'y': row.y})
    for team in sorted({p['team'] for p in points} - {'Unconfirmed','unknown','',None}):
        outfield = [p for p in points if p['team'] == team and
                    p['position_group'] != 'goalkeeper' and
                    not any(s in p['role'].lower() for s in ('goalkeeper','referee','keeper'))]
        if len(outfield) < 3:
            continue
        xy = np.array([[p['x'],p['y']] for p in outfield])
        hull, area = [], None
        try:
            shape = ConvexHull(xy)
            hull = xy[shape.vertices].tolist()
            area = float(shape.volume)
        except QhullError:
            pass
        # An instantaneous line estimate is only offered for ten identified
        # outfield roles and a known, consistent attack direction.
        directions = {people[p['player_id']].get('stadium_direction',people[p['player_id']].get('direction')) for p in outfield}
        bands = None
        if len(outfield)==10 and len(directions)==1 and directions <= {'right','left'} and all(p['position_group'] for p in outfield):
            xs = np.sort(xy[:,0] if 'right' in directions else 105-xy[:,0])
            groups = np.split(xs,np.where(np.diff(xs)>6)[0]+1)
            if 2<=len(groups)<=5 and all(len(g)<=5 for g in groups):
                bands = '-'.join(str(len(g)) for g in groups)
        teams.append({'team': team, 'visible_outfield': len(xy), 'line_estimate': bands,
                      'centroid': xy.mean(0).tolist(), 'length_m': float(np.ptp(xy[:,0])),
                      'width_m': float(np.ptp(xy[:,1])), 'hull_m2': area, 'hull': hull})
    return {'time_s': float(closest.time_s), 'players': points, 'teams': teams,
            'note': 'Visible players at one sampled frame in stadium orientation. Width, length and area describe this visible subset; they are not full-team formation or possession measurements. Confirm teams and goalkeeper roles first.'}


def occupancy(identifier):
    """GPU histogram and Gaussian smoothing of observed team presence."""
    import torch
    from .runtime import torch_device
    manifest=S.read_json(S.dataset_dir(identifier)/'manifest.json')
    q=stadium_points(S.load_tracks(identifier),manifest)
    result={'teams':[],'difference':None,'device':None,
            'note':'Normalised observed occupancy in common stadium orientation, smoothed by one grid cell. Differences reflect observed presence, not possession, pitch control or unobserved players.'}
    if q.empty:return result
    people={str(p['player_id']):p for p in manifest['players']}
    q['team']=q.player_id.map(lambda p:people[p].get('team','Unconfirmed'))
    q=q[~q.team.isin(['Unconfirmed','unknown','',None])]
    q=q[~q.player_id.map(lambda p: 'referee' in (people[p].get('role') or '').lower())]
    if q.empty:return result
    device=torch_device();result['device']=str(device)
    with torch.inference_mode():
        axis=torch.arange(-3,4,device=device,dtype=torch.float32)
        kernel=torch.exp(-(axis[:,None].square()+axis[None,:].square())/2)
        kernel=(kernel/kernel.sum())[None,None]
        for team,group in q.groupby('team'):
            xy=torch.tensor(group[['x','y']].to_numpy(),dtype=torch.float32,device=device)
            x=(xy[:,0]/105*32).long().clamp(0,31);y=(xy[:,1]/68*20).long().clamp(0,19)
            hist=torch.bincount(y*32+x,minlength=640).float().reshape(1,1,20,32)
            smooth=torch.nn.functional.conv2d(hist,kernel,padding=3)[0,0]
            smooth/=smooth.sum()
            result['teams'].append({'team':team,'observations':len(group),'heatmap':smooth.cpu().tolist()})
    if len(result['teams'])==2:
        a,b=result['teams']
        result['difference']={'positive_team':a['team'],'negative_team':b['team'],
                              'heatmap':(np.array(a['heatmap'])-np.array(b['heatmap'])).tolist()}
    return result


def player_analysis(identifier, pid):
    directory = S.dataset_dir(identifier)
    manifest = S.read_json(directory/'manifest.json')
    if not any(str(p['player_id']) == pid for p in manifest['players']):
        raise FileNotFoundError('Unknown player')
    tracks = S.load_tracks(identifier)
    kit = S.read_json(directory/'kit_groups.json', {})
    return S.clean_json({'motion': motion(tracks[tracks.player_id.eq(pid)], manifest['sampling_hz']),
                         'kit_suggestion': kit.get('tracks', {}).get(pid), 'kit_note': kit.get('note')})


def mot_export(identifier):
    """Source-frame MOTChallenge export with ID map and sampling metadata."""
    directory = S.dataset_dir(identifier)
    manifest = S.read_json(directory/'manifest.json')
    tracks = S.load_tracks(identifier)
    columns = ['bbox_x','bbox_y','bbox_w','bbox_h']
    if not all(c in tracks for c in columns):
        raise ValueError('MOT export requires image bounding boxes; this source has pitch coordinates only')
    t = tracks[tracks.detected.eq(1)].copy()
    t = t[np.isfinite(t[columns]).all(axis=1) & t.bbox_w.gt(0) & t.bbox_h.gt(0)]
    ids = {pid: i+1 for i,pid in enumerate(sorted(t.track_id.unique()))}
    out = io.StringIO()
    for row in t.sort_values(['frame','track_id']).itertuples():
        frame = int(getattr(row,'source_frame',row.frame)) + 1
        score = float(getattr(row,'confidence',1))
        out.write(f'{frame},{ids[row.track_id]},{row.bbox_x:.3f},{row.bbox_y:.3f},{row.bbox_w:.3f},{row.bbox_h:.3f},{score:.6f},-1,-1,-1\n')
    import json
    metadata = {'dataset_id': identifier, 'track_ids': ids,
                'frame_clock': 'one-based source video frames' if 'source_frame' in t else 'one-based imported frames',
                'source_fps': manifest.get('source_fps'), 'sampling_hz': manifest['sampling_hz'],
                'source_offset_s': manifest.get('source_offset_s',0),
                'note': 'Predictions only, not benchmark results. Evaluate against matching ground truth using the same sampled frames and person classes. Missing source frames are not predicted.'}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(identifier+'.txt',out.getvalue())
        archive.writestr('metadata.json',json.dumps(metadata,indent=2))
    return buffer.getvalue()
