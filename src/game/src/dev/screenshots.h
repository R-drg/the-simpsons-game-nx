// Development: save the presented frame periodically, whichever graphics
// system draws it (the emulated GPU or the Switch native renderer), so a
// headless PC run can be compared against the reference renderer. PC dev
// builds only (SIMPSONS_NATIVE_RENDERER); the Switch uses nx_screenshot_*.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/system/interfaces/graphics.h>
#include <rex/ui/presenter.h>

REXCVAR_DEFINE_UINT32(dev_screenshot_interval_ms, 0, "Dev",
                      "Save the presented frame as <dev_screenshot_prefix>_N.ppm this often");
REXCVAR_DEFINE_STRING(dev_screenshot_prefix, "dev_shot", "Dev",
                      "Path prefix of the frames dev_screenshot_interval_ms saves");

namespace simpsons::dev {

class Screenshots {
 public:
  ~Screenshots() { Stop(); }

  void Start(rex::system::IGraphicsSystem* graphics) {
    const uint32_t interval_ms = REXCVAR_GET(dev_screenshot_interval_ms);
    if (!interval_ms || !graphics || !graphics->presenter() || thread_.joinable()) return;
    running_ = true;
    thread_ = std::thread([this, graphics, interval_ms]() {
      const std::string prefix = REXCVAR_GET(dev_screenshot_prefix);
      uint32_t index = 0;
      auto next = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_ms);
      while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (std::chrono::steady_clock::now() < next) continue;
        next += std::chrono::milliseconds(interval_ms);
        rex::ui::RawImage image;
        if (!graphics->presenter()->CaptureGuestOutput(image) || !image.width) continue;
        const std::string path = fmt::format("{}_{:04}.ppm", prefix, index++);
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) continue;
        std::fprintf(file, "P6\n%u %u\n255\n", image.width, image.height);
        std::vector<uint8_t> row(size_t(image.width) * 3);
        for (uint32_t y = 0; y < image.height; ++y) {
          const uint8_t* source = image.data.data() + image.stride * y;
          for (uint32_t x = 0; x < image.width; ++x) std::memcpy(&row[x * 3], source + x * 4, 3);
          std::fwrite(row.data(), 1, row.size(), file);
        }
        std::fclose(file);
        REXLOG_INFO("dev screenshot: {}", path);
      }
    });
  }

  void Stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }

 private:
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace simpsons::dev
