import io
import zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from football_profiler import analytics as A, storage as S, soccernet as SN


def rows(times, x, periods=None):
    n=len(times)
    return pd.DataFrame({'frame':range(n),'time_s':times,'x':x,'y':10.,'detected':1,
                         'calibration_valid':1,'period':periods or [1]*n,'player_id':'p','track_id':'p'})


def test_motion_excludes_gaps_cuts_estimates_and_teleports():
    t=rows([0,.1,.2,.5,.6,.7,.8,.9],[0,.1,.2,.5,.6,90,90.1,90.2],[1,1,1,1,2,2,2,2])
    t.loc[6,'detected']=0
    result=A.motion(t,10)
    assert result['distance_m']==pytest.approx(.2)
    assert result['observed_motion_seconds']==pytest.approx(.2)
    assert result['valid_steps']==2
    assert sum(z['seconds'] for z in result['zones'])==pytest.approx(.2)
    assert result['mean_speed_kmh']==pytest.approx(3.6)


def test_unmapped_motion_is_unknown_not_zero_distance():
    t=rows([0,.1],[np.nan,np.nan])
    result=A.motion(t,10)
    assert result['distance_m'] is None
    assert result['peak_speed_kmh'] is None
    assert result['observed_motion_seconds']==0


def test_stadium_orientation_reverses_confirmed_left_only():
    t=rows([0,0],[10.,15.]);t['player_id']=['left','right']
    manifest={'source_kind':'model_predictions','players':[{'player_id':'left','direction':'left'},
                                                           {'player_id':'right','direction':'right'}]}
    result=A.stadium_points(t,manifest)
    assert result.x.tolist()==[95.,15.]
    assert result.y.tolist()==[58.,10.]
    # Old provider data normalised per team cannot silently be overlaid.
    manifest['source_kind']='provider_tracking'
    assert A.stadium_points(t,manifest).empty


def test_shape_excludes_goalkeeper_and_uses_simultaneous_frame():
    t=rows([0]*4+[10],[10,20,20,1,100]);t['player_id']=['a','b','c','g','a'];t['frame']=[0,0,0,0,100]
    t.loc[2,'y']=20
    manifest={'source_kind':'reference_annotations','sampling_hz':10,
              'players':[{'player_id':p,'team':'A','role':'Goalkeeper' if p=='g' else 'Player'} for p in ('a','b','c','g')]}
    result=A.snapshot(t,manifest,0)
    assert result['teams'][0]['visible_outfield']==3
    assert result['teams'][0]['length_m']==10
    assert result['teams'][0]['hull_m2']==pytest.approx(50)
    assert not A.snapshot(t,manifest,5)['players']


def test_library_cannot_accept_arbitrary_path(client,tmp_path,monkeypatch):
    monkeypatch.setenv('PITCHPROFILE_SOCCERNET',str(tmp_path/'library'))
    assert client.get('/api/soccernet/library').json()['videos']==[]
    assert client.post('/api/soccernet/analyse',json={'library_id':'../../secret'}).status_code in (404,405)
    assert client.post('/api/upload').status_code in (404,405)


def test_soccernet_event_clock_uses_position_and_half(tmp_path,monkeypatch):
    monkeypatch.setenv('PITCHPROFILE_SOCCERNET',str(tmp_path))
    S.write_json(tmp_path/'labels/game/Labels-v2.json',{'annotations':[
        {'gameTime':'1 - 01:00','position':'60000','label':'Goal'},
        {'gameTime':'2 - 46:00','position':'61000','label':'Foul'},
        {'gameTime':'2 - 45:00','position':'nan','label':'bad'}]})
    assert SN.annotations('game',2)[0]['time_s']==61
    assert len(SN.annotations('game',2))==1
    with pytest.raises(ValueError):SN.annotations('../../outside',1)


def test_soccernet_validation_name_preserves_holdout_contract(tmp_path,monkeypatch):
    monkeypatch.setenv('PITCHPROFILE_SOCCERNET',str(tmp_path))
    S.write_json(tmp_path/'official_splits.json',{'games':{'a':'train','b':'val','c':'test'}})
    assert SN.official_splits()['games']=={'a':'train','b':'validation','c':'test'}


def test_external_video_frame_and_calibration_use_source_offset(client,sample_dataset):
    identifier,_,m,_=sample_dataset
    m.update(source_kind='model_predictions',video='input.mkv',source_offset_s=45)
    S.write_json(S.dataset_dir(identifier)/'manifest.json',m)
    with patch('football_profiler.vision.read_frame',return_value=np.zeros((20,20,3),np.uint8)) as read:
        assert client.get(f'/api/datasets/{identifier}/frame?seconds=0.2').status_code==200
        assert read.call_args.args[1]==pytest.approx(45.2)


def test_mot_keeps_source_frame_indices_and_no_track_collisions(sample_dataset):
    identifier,_,m,t=sample_dataset
    for col,value in [('bbox_x',3),('bbox_y',4),('bbox_w',10),('bbox_h',20),('confidence',.8)]:t[col]=value
    t['source_frame']=np.arange(10)*5+100
    t.loc[5:,'track_id']='s2-t1'
    S.save_tracks(identifier,t)
    with zipfile.ZipFile(io.BytesIO(A.mot_export(identifier))) as z:
        lines=z.read(identifier+'.txt').decode().splitlines()
        assert lines[0].split(',')[0]=='101'
        assert lines[1].split(',')[0]=='106'
        assert lines[0].split(',')[1]!=lines[5].split(',')[1]


def test_kit_clustering_uses_gpu_and_never_assigns_semantic_team():
    from football_profiler.kits import group_colours
    result=group_colours({'a':[[100,180,150]]*8,'b':[[200,120,110]]*8,'c':[[30,128,128]]*8})
    assert result['device'].startswith('cuda')
    assert len(result['groups'])==3
    assert result['tracks']['a']['vote_share']==1
    assert 'team' not in result['tracks']['a']


def test_occupancy_uses_common_pitch_and_gpu(sample_dataset):
    identifier,_,m,t=sample_dataset
    m['coordinate_orientation']='stadium'
    S.write_json(S.dataset_dir(identifier)/'manifest.json',m)
    result=A.occupancy(identifier)
    assert result['device'].startswith('cuda')
    assert np.array(result['teams'][0]['heatmap']).sum()==pytest.approx(1)


def test_converter_clips_boxes_and_keeps_empty_frames(tmp_path):
    from scripts.prepare_detector_data import convert_sequence
    seq=tmp_path/'SNMOT-001';(seq/'gt').mkdir(parents=True);(seq/'img1').mkdir()
    (seq/'seqinfo.ini').write_text('[Sequence]\nimWidth=100\nimHeight=80\nimDir=img1\n')
    (seq/'gt/gt.txt').write_text('1,1,-9,1,20,20,1,1,1\n1,2,10,10,5,5,1,2,1\n')
    for name in ('000001.jpg','000002.jpg'):(seq/'img1'/name).write_bytes(b'fixture')
    out=tmp_path/'yolo'
    assert convert_sequence(seq,out,'train',{1})==2
    values=(out/'labels/train/SNMOT-001_000001.txt').read_text().split()
    assert list(map(float,values[1:]))==pytest.approx([.05,.125,.1,.25])
    assert not (out/'labels/train/SNMOT-001_000002.txt').read_text()


def test_official_tracking_evaluator_on_artificial_perfect_tracks(tmp_path):
    pytest.importorskip('trackeval')
    from scripts.evaluate_tracking import evaluate
    gt=tmp_path/'gt';seq=gt/'fixture';(seq/'gt').mkdir(parents=True)
    (seq/'seqinfo.ini').write_text('[Sequence]\nname=fixture\nimWidth=100\nimHeight=100\nseqLength=2\nframeRate=25\n')
    (seq/'gt/gt.txt').write_text('1,1,10,10,20,20,1,1,1\n2,1,11,10,20,20,1,1,1\n')
    pred=tmp_path/'pred';pred.mkdir()
    (pred/'fixture.txt').write_text('1,1,10,10,20,20,1,-1,-1,-1\n2,1,11,10,20,20,1,-1,-1,-1\n')
    result=evaluate(gt,pred,tmp_path/'scores')
    metrics=result['results']['MotChallenge2DBox']['pred']['COMBINED_SEQ']['pedestrian']
    assert metrics['Identity']['IDF1']==pytest.approx(1)
    assert np.asarray(metrics['HOTA']['HOTA']).mean()==pytest.approx(1)
