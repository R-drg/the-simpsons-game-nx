# Patches against the ReXGlue SDK

The port builds on the ReXGlue SDK vendored in TheSimpsonsGameRecomp at `tools/rexglue-sdk`, commit
`285bea3259ce194c28f9e9ef01ff6897e4893734` (SDK 0.8 with that project's changes). Apply the patches in the order of
`series`, from the root of a TheSimpsonsGameRecomp checkout at that commit:

```sh
while read p; do git apply --directory=tools/rexglue-sdk "/path/to/patches/rexglue-sdk/$p"; done < series
```

- `0001-switch-port.patch`: everything the Switch build needs, most of it adapted from project8-sw's SDK patches:
  - the Horizon platform layer: guest memory, threads and fibers, exceptions and crash logs, the window and swap
    chain on `nn::vi`, audio through audout, input through HID, the CMake support for building the NRO
    (`cmake/rexglue_switch.cmake`) and FFmpeg's configuration for the Switch;
  - what the native renderer uses from the SDK: the graphics interface it replaces, ring progress and read-pointer
    write-back, GPU timestamps, implicit-LOD texture sampling (`gpu_implicit_lod_sampling`), and an observer on the
    shader interpreter's vertex fetches, used to work out which vertices a draw reads;
  - performance: the game's threads are spread over the three cores the process gets, and a profiler that writes
    `logs/rex/rex_perfil.log` (thread CPU, clocks, audio) every ten seconds.

The changes are under the SDK's BSD-3-Clause licence, as the SDK itself.
