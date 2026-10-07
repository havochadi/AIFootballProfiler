"""Regression checks for the ffmpeg lookup and the portable overlay encoding."""
import subprocess
import unittest
from unittest.mock import patch


from football_profiler import vision


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


if __name__=="__main__":
    unittest.main()
