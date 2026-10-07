# PitchProfile slide deck

Source of the 13-slide presentation: cover, what it does, app features, the whole pipeline, how
raw video becomes statistics (data at each step, then three worked examples), the models (a map of
all of them, then deep dives on seeing, identity and action spotting), accuracy, difficulties and
next steps. Each file in `slides/` is one slide in the Claude Slides format (1920×1080 HTML
sections with inline styles, speaker notes in the `<aside>`); `deck.json` holds the slide order,
sections and fonts (Space Grotesk, IBM Plex Sans).

The files only render in the Slides viewer, not as plain web pages. All figures come from the
project docs (`PitchProfile/FULL_MATCH_ANALYSIS.md`, `MODELS.md`, `DETECTION_RESEARCH.md`) and
the demo data in `PitchProfileDemo/`.

Colour code on the pipeline and model slides: blue = pretrained and used as released, green =
fine-tuned or trained in this project, grey = not a learned model.

The cover still carries a placeholder: `[Team name · members]`.
