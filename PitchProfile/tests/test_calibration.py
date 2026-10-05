"""Regression checks for calibration and portable tracking-overlay output."""
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from football_profiler import storage as S
from football_profiler import vision


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        data_patch=patch.object(S,"DATA",Path(self.temporary.name))
        data_patch.start()
        self.addCleanup(data_patch.stop)
        self.identifier="calibration-test"
        self.directory=S.dataset_dir(self.identifier,True)
        self.manifest={
            "id":self.identifier,"source_kind":"model_predictions","source":"Test clip",
            "video":"input.mp4","sampling_hz":1,"duration_seconds":2,"cut_times":[],
            "players":[
                {"player_id":"left","direction":"left","direction_known":True,"eligible_frames":2},
                {"player_id":"right","direction":"right","direction_known":True,"eligible_frames":2},
            ],
        }
        S.write_json(self.directory/"manifest.json",self.manifest)
        rows=[]
        for frame in (0,1):
            for player,x,y in (("left",85,38),("right",20,30)):
                rows.append({"frame":frame,"time_s":float(frame),"player_id":player,
                             "track_id":player,"period":1,"x":x,"y":y,"detected":1,
                             "calibration_valid":1,"bbox_x":15,"bbox_y":20,
                             "bbox_w":10,"bbox_h":10})
        S.save_tracks(self.identifier,pd.DataFrame(rows))
        self.landmarks=[[0,0],[105,0],[105,68],[0,68]]

    def calibrate(self,**kwargs):
        with patch.object(vision,"read_frame",return_value=np.zeros((80,120,3),np.uint8)), \
             patch.object(vision.cv2,"VideoCapture"):
            return vision.calibrate(self.identifier,self.landmarks,self.landmarks,
                                    static_camera=True,**kwargs)

    def test_recalibration_preserves_left_direction_and_unselected_interval(self):
        self.calibrate(end_s=1)
        tracks=S.load_tracks(self.identifier)
        left=tracks[tracks.player_id.eq("left")]
        right=tracks[tracks.player_id.eq("right")]
        np.testing.assert_allclose(left[["x","y"]],[[85,38],[85,38]],atol=1e-4)
        np.testing.assert_allclose(right[["x","y"]],[[20,30],[20,30]],atol=1e-4)
        self.calibrate(end_s=1)
        repeated=S.load_tracks(self.identifier)
        np.testing.assert_allclose(repeated[["x","y"]],tracks[["x","y"]],atol=1e-4)

    def test_camera_cut_rejection_does_not_change_saved_tracks(self):
        self.manifest["cut_times"]=[1]
        S.write_json(self.directory/"manifest.json",self.manifest)
        before=(self.directory/"tracks.csv.gz").read_bytes()
        for static in (False,True):
            with self.subTest(static=static),self.assertRaisesRegex(ValueError,"camera cut"):
                vision.calibrate(self.identifier,self.landmarks,self.landmarks,static_camera=static)
        self.assertEqual(before,(self.directory/"tracks.csv.gz").read_bytes())
        self.assertNotIn("calibration",S.read_json(self.directory/"manifest.json"))

    def test_calibration_can_end_or_start_at_a_camera_cut(self):
        self.manifest["cut_times"]=[1]
        S.write_json(self.directory/"manifest.json",self.manifest)
        self.assertEqual(self.calibrate(end_s=1)["valid_sampled_frames"],1)
        self.assertEqual(self.calibrate(start_s=1,reference_s=1,end_s=2)["valid_sampled_frames"],1)


class OverlayTests(unittest.TestCase):
    def test_system_ffmpeg_takes_precedence(self):
        with patch.object(vision.shutil,"which",return_value="system-ffmpeg"):
            self.assertEqual(vision.ffmpeg_executable(),"system-ffmpeg")

    def test_bundled_ffmpeg_is_used_without_system_install(self):
        with patch.object(vision.shutil,"which",return_value=None), \
             patch("imageio_ffmpeg.get_ffmpeg_exe",return_value="bundled-ffmpeg"):
            self.assertEqual(vision.ffmpeg_executable(),"bundled-ffmpeg")

    def test_unavailable_ffmpeg_is_reported(self):
        with patch.object(vision.shutil,"which",return_value=None), \
             patch("imageio_ffmpeg.get_ffmpeg_exe",side_effect=RuntimeError("unavailable")):
            self.assertIsNone(vision.ffmpeg_executable())

    def test_gpu_overlay_uses_nvenc_and_cpu_requires_explicit_device(self):
        with patch.object(vision,"ffmpeg_executable",return_value="ffmpeg"), \
             patch.object(vision.subprocess,"run") as run:
            self.assertEqual(vision.encode_overlay("source.avi","overlay.mp4","cuda:0"),"h264_nvenc")
            command=run.call_args.args[0]
            self.assertIn("h264_nvenc",command)
            self.assertNotIn("libx264",command)
            self.assertEqual(vision.encode_overlay("source.avi","overlay.mp4","cpu"),"libx264")
            self.assertIn("libx264",run.call_args.args[0])

    def test_gpu_encoding_failure_never_retries_on_cpu(self):
        error=subprocess.CalledProcessError(1,["ffmpeg"],stderr=b"NVENC unavailable")
        with patch.object(vision,"ffmpeg_executable",return_value="ffmpeg"), \
             patch.object(vision.subprocess,"run",side_effect=error) as run:
            with self.assertRaisesRegex(ValueError,"GPU overlay encoding failed"):
                vision.encode_overlay("source.avi","overlay.mp4","cuda:0")
            run.assert_called_once()

    def test_failed_video_writer_fails_job_and_releases_resources(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(S,"DATA",Path(temporary)):
            capture=MagicMock()
            capture.read.return_value=(True,np.zeros((80,120,3),np.uint8))
            writer=MagicMock()
            writer.isOpened.return_value=False
            model=MagicMock()
            model.track.return_value[0].boxes.id=None
            info={"fps":1,"frames":1,"width":120,"height":80,"duration":1}
            with patch.object(vision,"video_info",return_value=info), \
                 patch.object(vision,"torch_device",return_value="cuda:0"), \
                 patch.object(vision,"detector",return_value=model), \
                 patch.object(vision.cv2,"VideoCapture",return_value=capture), \
                 patch.object(vision.cv2,"VideoWriter",return_value=writer):
                with self.assertRaisesRegex(ValueError,"overlay could not be written"):
                    vision.process_video("writer-test",Path("input.mp4"),"Test")
            capture.release.assert_called_once()
            writer.release.assert_called_once()
            self.assertFalse((Path(temporary)/"datasets/writer-test/manifest.json").exists())


if __name__=="__main__":
    unittest.main()
