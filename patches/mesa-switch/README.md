# Patches against mesa-switch

The Switch build links NVK from https://github.com/danfromtico/mesa-switch at
commit `d4a00ea0ab3` (Mesa 26.2.3). These patches apply on top of it.

- `0001-nvk-switch-resolve-set-ubo-descriptors.patch`: before Turing, NVK binds
  a uniform buffer whose descriptor lives in a regular descriptor set through
  an indirect GPFIFO segment, which costs two GPFIFO entries and a
  no-prefetch front-end stall per binding. At about 1,200 draws a frame in
  Free Skate that nearly filled the 2,048-entry GPFIFO: every kickoff stalled
  on "native submission pressure" and the device was eventually lost. The
  patch reads the descriptor from the set's CPU mapping at bind time, which
  Vulkan permits for sets without update-after-bind. On hardware it removed
  the pressure warnings and the crash, and raised Free Skate from 2-3 to 7-8
  fps.
- `0002-nak-gm20b-latency-and-flattening.patch`: two NAK scheduling changes
  for pre-Volta GPUs taken from nfsmw-nx's Mesa patch (MIT, see
  https://github.com/StevensND/nfsmw-nx/tree/main/mesa): texture and global
  memory latency estimated at 200 cycles instead of 32, and
  `nir_opt_peephole_select` limit 8 instead of 0. It also adds a Switch
  revision bit to the compiler flags so pipeline caches built by an earlier
  driver are recompiled rather than reused.
- `0003-nvk-switch-direct-draws-and-memory-query.patch`: two CPU costs of
  the driver on the Switch. Before Pascal every draw goes through a draw
  macro that writes four draw parameters into a constant buffer; when they
  are all zero and the previous draw of the command buffer left them zero,
  the draw is issued directly (`BEGIN`, draw, `END`), as the nouveau GL
  driver does on these GPUs. And `os_get_available_system_memory` walked the
  whole newlib heap through `mallinfo` under its lock each time the memory
  allocator asked for its budget, which it does every few dozen allocations;
  the value is now reused for up to 100 ms.
- `0004-switch-build-script.patch`: the script that builds the driver
  archives this port links (`build-switch-p8.sh`, run from the mesa-switch
  checkout), the Meson native file it uses, and a fix to the Rust compiler
  wrapper: procedural macros run inside the compiler on the build machine, so
  they must never be cross-compiled for the Switch.

Apply them in order with `git apply` on a checkout of that commit, then run
`./build-switch-p8.sh`. It builds the driver's static libraries in
`builddir-switch/`. Merge them into one `lib/libvulkan.a`, check and index it
with `tools/switch/stage-mesa-archive.py`, and put `libnak_rs.a` next to it;
`REXGLUE_SWITCH_NVK_SDK` points at the folder that holds `lib/`.
