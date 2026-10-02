# Building the port yourself

## In short

Players do not need this page: the [installer page](https://r-drg.github.io/the-simpsons-game-nx-installer/) makes
the package from their own copy of the game, with the released NRO. This page is for building the NRO (the Switch
program) yourself, from the source code in this repository.

The steps, in order: prepare the SDK, build the graphics driver, translate the game to C++, build the NRO, and copy
everything to the SD card. The release was built on Linux, inside Docker, with exactly these steps.

## What you need

- Docker. The Switch toolchain runs in an image built from `tools/switch/Dockerfile`: devkitPro's devkitA64 image
  (devkitA64, libnx and the portlibs) plus clang 19 and glslang.
- Git, CMake 3.25 or newer, Ninja, Python 3, and a host C++ compiler (clang) to build the code generator.
- Your own copy of The Simpsons Game for Xbox 360, extracted, with the supported `default.xex`
  (SHA-256 `71d99dad06be1b512fc3058123b84fdad71339205a7e9249058ac5e34a82a231`). Nothing from the game is in this
  repository, and nothing made from it may be added to it: `tools/check_no_game_content.sh` refuses it.
- About 30 GB of RAM for a parallel build (each translated file takes up to 3 GB in GCC), or fewer jobs with less.
  A full build takes about 40 minutes at 8 jobs.

```sh
docker build -t simpsons-switch tools/switch
```

## 1. Prepare the SDK

**Why:** the port runs on the ReXGlue SDK as TheSimpsonsGameRecomp ships it (0.8 with that project's changes), plus
this port's changes: the Horizon (Switch) runtime, memory, input, audio, crash handling, and the GPU work the native
renderer reuses (the shader translator, the shader interpreter).

```sh
git clone https://github.com/YesterMester/TheSimpsonsGameRecomp base
git -C base checkout 285bea3259ce194c28f9e9ef01ff6897e4893734
while read p; do git -C base apply --directory=tools/rexglue-sdk "$PWD/patches/rexglue-sdk/$p"; done < patches/rexglue-sdk/series
```

The patched SDK is `base/tools/rexglue-sdk`, the `REXSDK_DIR` of step 4.

## 2. Build the graphics driver

**Why:** Vulkan on the Switch is NVK, Mesa's driver for NVIDIA GPUs, ported to Horizon by mesa-switch. It is linked
into the NRO, and this port changes it (see [patches/mesa-switch/README.md](../patches/mesa-switch/README.md)).

```sh
git clone https://github.com/danfromtico/mesa-switch mesa
git -C mesa checkout d4a00ea0ab3
for p in patches/mesa-switch/0*.patch; do git -C mesa apply "$PWD/$p"; done
cd mesa && ./build-switch-p8.sh
```

The script builds its own Docker image (Rust and Meson for an AArch64 Horizon target) and leaves the driver's static
libraries in `mesa/builddir-switch/`. Merge them into one `lib/libvulkan.a`, check and index it with
`tools/switch/stage-mesa-archive.py`, and put `libnak_rs.a` next to it. That folder (the one holding `lib/`) is the
"driver SDK" of step 4.

## 3. Translate the game to C++

**Why:** static recompilation reads the game's PowerPC code from `default.xex` and writes it out as C++. The
configuration that says how (addresses, function starts, patches) is `src/game/simpsons_manifest.toml` and
`src/game/functions_patch.toml`; the output is a translation of your copy of the game and must never be published.

Build the SDK once for your PC to get its `rexglue` tool (TheSimpsonsGameRecomp's README explains it), then:

```sh
cp src/game/simpsons_manifest.toml src/game/simpsons_manifest.local.toml
# In the copy, replace REPLACE_ME with the folder of your extracted disc.
rexglue codegen src/game/simpsons_manifest.local.toml
```

It writes `src/game/generated/default/`, which git ignores. The release was built from the translation at
TheSimpsonsGameRecomp commit `285bea32`, unchanged.

## 4. Build the NRO

```sh
docker run --rm -v "$PWD:/project" -v /path/to/base:/base -v /path/to/driver-sdk:/mesa -w /project \
  simpsons-switch sh -c '
    cmake -S src/game -B build/switch -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=/project/tools/switch/toolchain.cmake \
      -DREXSDK_DIR=/base/tools/rexglue-sdk -DREXGLUE_SWITCH_NVK_SDK=/mesa &&
    cmake --build build/switch -j8'
```

The result is `build/switch/simpsons.nro`, published as `the_simpsons_game.nro`, and next to it the unstripped
`build/switch/simpsons`, which loads at address 0, so `addr2line` takes the offsets of a crash log directly.

The build also compiles the replacement shaders in `tools/native-renderer/native_shaders/` with glslang and embeds
them in the NRO.

The SDK's FFmpeg and libmspack (LGPL-2.1) are linked statically. To relink with a modified version of either, change
it in the SDK tree and run this step again.

## 5. Copy it to the SD card

```text
/switch/the_simpsons_game/
  the_simpsons_game.nro     build/switch/simpsons.nro, renamed
  simpsons.toml             release/simpsons.toml
  LICENSES.txt              NOTICE
  game/                     your extracted disc: default.xex and its folders
```

Copy with a tool that checks what it wrote (DBI over MTP works). A `default.xex` damaged on the card makes the game
crash at start-up, reading a pointer from the wrong data.

## Before publishing anything

```sh
tools/check_no_game_content.sh
tools/make_notice.py --sdk /path/to/base/tools/rexglue-sdk > NOTICE
```
