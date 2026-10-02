# Known issues

## Performance

- **Clocks below the tested ones.** The release was tested with sys-clk at CPU 1785 MHz and GPU 768 MHz in handheld
  mode. On a test run where sys-clk was changed mid-game, the console fell to stock handheld clocks (CPU 1020, GPU
  307) for about 20 seconds during a level's opening cutscene. The cutscene ran at 19 FPS, its GPU time rose from
  about 12 to 31 ms a frame, and the cutscene's audio ran ahead of the picture: the audio is streamed in real time
  while the picture is drawn frame by frame. With the GPU then at 614 MHz, the rest of the level ran at 22–28 FPS.
  The port does not change the clocks itself (`switch_gpu_mhz = 0`), so keep sys-clk at the tested profile for the
  whole session.
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
