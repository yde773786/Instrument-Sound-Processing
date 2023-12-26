## Audio Segmentation
Primary goals of this project included being able to:
- Denoise an audio recording
- Apply effects to certain instruments in a recording involving multiple other artifacts/sounds.

In order to do the above, **audio segmentation** is essential. We need to _segment_ the audio such that we could specifically target a specific feature of the recording without affecting the rest. In [stft_denoising.ipynb](./Denoising/stft_denoising.ipynb), the experimentation on de-noising attempts to remove noise in the fourier domain (as noise should have lesser magnitude). However, the audio recording on performing inverse fourier transform contains unwanted artifacts of its own, and is unclear.

Why was de-noising through this approach a sub-par approach? The answer appears to be the loss of phase information, as well as perfection of time/frequency resolution when working in the frequency domain. Essentially, too much loss of information as we switch from and out of frequency domain. Arguably, more could be done to refine and decrease loss of information. Phase-reconstruction algorithms such as Griffin-Lim is one solution. However, and my take may be incorrect, but it feels like a means of damage control more than anything. The loss of information is too much to and from two major parts of the pipeline. Moreover, this only targets objective 1 (Denoise an audio recording).

Inspired by the success in Image segmentation, there are ideas revolving around the concept of dealing with Audio segmentation through the time-domain rather than extracting features in the frequency domain. The latter is helpful for classification, sure, but not generation, which is what we need.

Specifically, [Wave-U-Net](https://arxiv.org/pdf/1806.03185.pdf) appears to be a good potential approach for this problem.
