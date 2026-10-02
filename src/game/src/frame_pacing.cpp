// Menu frame pacing.
//
// The game shipped at 30 fps: its frame scheduler waits two vblanks per frame.
// The 60 FPS patch (the "li r4,1" hand patch in sub_82867A48) makes that one
// vblank everywhere. Gameplay copes, since its logic runs on real time and
// physics has its own fix, but the front end runs its logic once per frame:
// at 60 fps menus scroll and repeat inputs twice as fast as they were made
// for, and the title screen, which costs a little more than one 60 Hz frame,
// alternates 17 and 33 ms frames. While no level is simulating, this puts the
// frame scheduler back on the console's 30 fps cadence.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ppc.h>

REXCVAR_DEFINE_INT32(menu_frame_rate, 30, "GPU",
                     "Frame rate of menus, the title screen and loading screens, where the game "
                     "runs its logic once per frame and was made for 30 fps (0 = same as gameplay)")
    .range(0, 240);

// The 60 fps patch halves the time a frame may take. The Switch's CPU cannot
// keep the game's simulation inside 16.7 ms, and an uneven 40-50 fps paces
// worse than the 30 the game was made for, so there gameplay keeps the
// console's cadence too.
#if REX_PLATFORM_SWITCH
#define SIMPSONS_GAMEPLAY_FRAME_RATE_DEFAULT 30
#else
#define SIMPSONS_GAMEPLAY_FRAME_RATE_DEFAULT 0
#endif
REXCVAR_DEFINE_INT32(gameplay_frame_rate, SIMPSONS_GAMEPLAY_FRAME_RATE_DEFAULT, "GPU",
                     "Frame rate cap while a level is running (0 = uncapped, the 60 fps patch)")
    .range(0, 240);

// Steps the Havok world by f1 seconds of frame time. Runs every frame while a
// level is live and never in the front end.
REX_EXTERN(__imp__sub_827A55C0);
REX_EXTERN(sub_827A55C0);

// Returns the time until the frame scheduler's next deadline. r3 = the
// scheduler, which holds the frame's start timebase at +136 and its periods
// per frame at +144, read on every call. The period is the game's own fixed
// refresh rate (59.94 Hz), whatever rate the guest vblank runs at.
REX_EXTERN(__imp__sub_82718710);
REX_EXTERN(sub_82718710);

namespace {

constexpr uint32_t kSchedulerPeriodsPerFrame = 144;
// The refresh rate (float) the scheduler divides the timebase frequency by.
constexpr uint32_t kSchedulerRefreshHz = 0x82CF0404;

// Without a physics step for this long, no level is running.
constexpr int64_t kMenuAfterNs = 750'000'000;

std::atomic<int64_t> g_last_level_step_ns{0};

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint32_t LoadBE32(const uint8_t* base, uint32_t address) {
  uint32_t value;
  std::memcpy(&value, base + address, sizeof(value));
  return rex::byte_swap(value);
}

void StoreBE32(uint8_t* base, uint32_t address, uint32_t value) {
  value = rex::byte_swap(value);
  std::memcpy(base + address, &value, sizeof(value));
}

}  // namespace

REX_FUNC(sub_827A55C0) {
  g_last_level_step_ns.store(NowNs(), std::memory_order_relaxed);
  __imp__sub_827A55C0(ctx, base);
}

REX_FUNC(sub_82718710) {
  // Only the game's main thread runs the scheduler.
  static uint32_t game_periods = 0;
  static uint32_t applied_periods = 0;
  static bool in_menu = false;

  int32_t menu_rate = REXCVAR_GET(menu_frame_rate);
  int32_t gameplay_rate = REXCVAR_GET(gameplay_frame_rate);
  uint32_t scheduler = ctx.r3.u32;
  if ((menu_rate > 0 || gameplay_rate > 0) && scheduler) {
    uint32_t address = scheduler + kSchedulerPeriodsPerFrame;
    uint32_t current = LoadBE32(base, address);
    // Whatever the game itself set, unless it is still the menu value.
    if (!game_periods || current != applied_periods) {
      game_periods = std::max<uint32_t>(current, 1);
    }
    int64_t last_step = g_last_level_step_ns.load(std::memory_order_relaxed);
    bool menu = !last_step || NowNs() - last_step > kMenuAfterNs;
    uint32_t refresh_bits = LoadBE32(base, kSchedulerRefreshHz);
    float refresh_hz;
    std::memcpy(&refresh_hz, &refresh_bits, sizeof(refresh_hz));
    if (!(refresh_hz >= 20.0f && refresh_hz <= 240.0f)) {
      refresh_hz = 59.94f;
    }
    auto periods_for = [&](int32_t rate) {
      if (rate <= 0) return game_periods;
      return std::max(game_periods, uint32_t(std::lround(double(refresh_hz) / double(rate))));
    };
    uint32_t wanted = periods_for(menu ? menu_rate : gameplay_rate);
    if (wanted != current) {
      StoreBE32(base, address, wanted);
    }
    applied_periods = wanted;
    if (menu != in_menu) {
      in_menu = menu;
      REXLOG_INFO("Frame pacing: {:.2f} fps, {}", double(refresh_hz) / double(wanted),
                  menu ? "no level running (menus)" : "level running");
    }
  }
  __imp__sub_82718710(ctx, base);
}
