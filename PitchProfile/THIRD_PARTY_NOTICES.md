# Third-party notices

## SoccerNet

Revision: `26e8e46f8258e306fdbf019540e1eda4221a863d`. Official split metadata and dataset format integration.

MIT License

Copyright (c) 2023 SoccerNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## football-player-tracking

Revision: `d25079981c9a063be8e5bf32933a3bee864a7e1e`. Adapted BoT-SORT configuration and torso sampling approach; remaining analytics are implemented locally.

MIT License

Copyright (c) 2026 Shaurya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


The MIT licences cover repository code and metadata. Match videos retain their SoccerNet dataset access terms. Existing Ultralytics usage and its licence remain applicable.

## Full-match video analysis models (downloaded, not bundled)

`scripts/fetch_football_models.py` downloads these to the data drive and verifies SHA-256.
None of their code or weights is copied into this repository.

| Component | Source and revision | Licence | Use here |
| --- | --- | --- | --- |
| Football player/goalkeeper/referee/ball detector and ball detector (YOLOv8x) | [roboflow/sports](https://github.com/roboflow/sports) `examples/soccer/setup.sh` weights | Repository MIT; YOLOv8 models are Ultralytics AGPL-3.0 | Detection in every sampled frame |
| No Bells, Just Whistles (NBJW) pitch keypoint network `SV_kp` | [mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) commit `bd993b31c2917096c23bb8aadf148314d17f8345`, release v1.0.0 | GPL-2.0 | External checkout imported at runtime for automatic pitch calibration; not vendored |
| Jersey legibility classifier and SoccerNet fine-tuned PARSeq | [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) published weights | Creative Commons Attribution-NonCommercial 3.0 | Reading shirt numbers to link players across camera cuts; non-commercial use only |
| PARSeq (`strhub`) model code | [baudm/parseq](https://github.com/baudm/parseq) commit `1902db043c029a7e03a3818c616c06600af574be` | Apache-2.0 | Installed with `--no-deps` to run the PARSeq weights |
| SoccerNet team ball-action spotting devkit and T-DEED baseline checkpoint | [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) commit `091fed2fc35c33f7489f3596958a2fe385e37d65`; checkpoint from its README (SHA-256 `ed7c558d…1c72a`) | GPL-3.0 | External checkout imported at runtime to spot tackles, blocks, headers, crosses, lofted passes and set pieces; not vendored |
| CLIP ViT-B/16 image encoder | OpenAI CLIP weights via timm `vit_base_patch16_clip_224.openai` | MIT (weights), timm Apache-2.0 | Backbone of the identity model (the older frozen-feature re-identification head was retired on 8 October 2026); the projection head on top is trained locally from this project's shirt-number readings |

The jersey weights' non-commercial licence applies to any use of this analysis beyond
research and teaching. SoccerNet and SoccerTrack datasets keep their own access terms.
Their labels are used to measure accuracy; they are never an input when footage is analysed.
(The shot classifier they once trained was retired on 8 October 2026.)
