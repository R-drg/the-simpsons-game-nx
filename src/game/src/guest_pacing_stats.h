// Counters of the game's frame pacing waits, read by the native renderer's
// slow-frame report. Nothing replaces The Simpsons Game's waits yet, so they
// stay at zero; the report still prints them so its format matches Project 8's.
#pragma once

#include <atomic>
#include <cstdint>

namespace simpsons::pacing {
// Time the render thread slept in the presentation-interval wait.
inline std::atomic<uint64_t> g_vblank_wait_ns{0};
// Late frames let through without waiting for the next vblank.
inline std::atomic<uint64_t> g_late_skips{0};
}  // namespace simpsons::pacing
