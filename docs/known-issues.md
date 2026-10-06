# Known issues

## Performance

- **Clocks below the tested ones.** The release was tested with sys-clk at CPU 1785 MHz and GPU 768 MHz in handheld
  mode. On a test run where sys-clk was changed mid-game, the console fell to stock handheld clocks (CPU 1020, GPU
  307) for about 20 seconds during a level's opening cutscene. The cutscene ran at 19 FPS, its GPU time rose from
  about 12 to 31 ms a frame, and the cutscene's audio ran ahead of the picture. With the GPU then at 614 MHz, the
  rest of the level ran at 22–28 FPS. The port does not change the clocks itself (`switch_gpu_mhz = 0`), so keep
  sys-clk at the tested profile for the whole session. Stock clocks were not tested again with 1.0.1.
- **Heavy particle scenes.** With many effects at once (Homer's burp, blowing things up) the frame rate can drop
  briefly to 26–28 FPS. The renderer then takes 7–14 ms a frame and the GPU 14–15 ms; what is left is the game's
  own CPU threads (the audio core thread up to 75% of a core, the main thread 45%).
- **First appearance of an effect.** A short hitch while its pipeline is compiled. The pipelines are cached on the
  SD card (`simpsons/cache`), so it happens once.

## Rendering

- Render targets of the game's 4x MSAA passes are drawn at 1x, at the size of the equivalent EDRAM area, which makes
  some of them larger than they need to be (up to 1280x2048). It has no visible effect and does not limit the frame
  rate.

## Not tested

- Docked mode, and stock clocks other than the case above.
- Releases of the game other than the USA retail disc: they do not run.

## Fixed in 1.0.1

- **Cutscenes slowed down and their audio got ahead of the picture.** The movie player keeps time with the audio
  stream and drops late pictures, but VP6 cannot skip a picture it has not decoded, and the game's recompiled VP6
  decoder took 92–98% of a core: the picture fell further and further behind. The cutscenes are now decoded by
  FFmpeg (`movie_decoder = "ffmpeg"`; `"check"` logs how the two decoders compare).
- **Drops to 24–27 FPS in the Springfield hub.** The main thread takes about 33 ms a frame there. The game's frame
  scheduler gave a frame that was already late up to 16.7 ms more of background streaming work, so a 35 ms frame
  became a 50 ms one. A late frame now gives that work at most `late_frame_task_ms` (2 ms); raise it if streaming
  falls behind (pop-in), or set -1 for the game's behaviour.
