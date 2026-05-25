# AED Non-Speech Probe

Notebook accompanying the [Hearing Things](https://pzulawski.github.io/machine-learning/ml/aed-hallucination-probe.html) series of posts, which investigates hallucinations in the [Cohere `cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) encoder-decoder ASR model.

## Contents

`cohere_speech_probe.ipynb` — the main analysis notebook, covering:

- **Encoder probe**: forward hooks collect mean-pooled residual stream activations at each encoder layer for speech and non-speech audio; a logistic regression trained per layer measures how linearly separable the two classes are. The probe finds near-perfect separation from the first layer onwards, establishing that the encoder builds a strong internal representation of the speech/non-speech distinction regardless of whether the model ultimately hallucates.
- **Decoder probe**: the same approach applied to the decoder, collecting per-token hidden states during generation. The probe is trained on hallucinating segments (non-speech audio that produced non-empty output) versus normal transcriptions, and evaluated on a held-out set as a runtime hallucination detector.

## Data format

The notebook expects segment lists in a whitespace-delimited format with columns `id`, `audio_path`, `start`, `end`, and an optional transcript. The `start` and `end` fields are offsets in seconds into the audio file, allowing segments to be indexed out of larger recordings. Set the `NON_SPEECH_AUDIO`, `SPEECH_AUDIO1`, and `SPEECH_AUDIO2` paths at the top of the data-loading cell before running.
