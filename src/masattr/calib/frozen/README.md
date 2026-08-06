# Frozen calibration maps

`masattr e0` writes `calibration.json` here: the per-type score→probability maps
fit **once** on paper 1's step-labeled single-agent corpus, plus the
first-crossing threshold chosen on that same corpus.

The file carries a `content_hash`, and `FrozenCalibration.load` refuses a file
whose hash no longer matches its contents — an in-place edit fails loudly
instead of silently moving every downstream number.

Nothing is committed here: the maps depend on which judge scored the corpus, so
they are a run artifact, not a repo artifact.
