// simpsons - ReXGlue Recompiled Project
//
// Customize your app by overriding virtual hooks from rex::ReXApp.

#pragma once

#include <rex/rex_app.h>
#include <rex/runtime.h>
#include <rex/system/xmemory.h>
#if REX_PLATFORM_SWITCH
#include <rex/filesystem.h>
#endif
#if REX_PLATFORM_SWITCH || SIMPSONS_NATIVE_RENDERER
#include <rex/runtime.h>

#include "native/native_system.h"
#endif
#if SIMPSONS_NATIVE_RENDERER && !REX_PLATFORM_SWITCH
#include "dev/screenshots.h"
#include "dev/script_input.h"
#endif

namespace simpsons {
// startup_check.cpp
void LogLoadedImage(const uint8_t* base);
}  // namespace simpsons

class SimpsonsApp : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(
      rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<SimpsonsApp>(new SimpsonsApp(ctx, "simpsons",
        PPCImageConfig));
  }

#if REX_PLATFORM_SWITCH
  void OnConfigurePaths(rex::PathConfig& paths) override {
    const auto directory = rex::filesystem::GetExecutableFolder();
    if (paths.game_data_root.empty()) paths.game_data_root = directory / "game";
  }
#endif

#if REX_PLATFORM_SWITCH || SIMPSONS_NATIVE_RENDERER
  // The SDK's emulated Xbox 360 GPU runs at a few frames a second on the
  // Switch (EDRAM emulation, and write watching that Horizon can only do by
  // unmapping pages); the native renderer draws the game's command stream
  // with Vulkan directly.
  void OnPreSetup(rex::RuntimeConfig& config) override {
    if (simpsons::native::Enabled()) {
      config.graphics = simpsons::native::CreateGraphicsSystem();
    }
#if SIMPSONS_NATIVE_RENDERER && !REX_PLATFORM_SWITCH
    // --input_json / --input_json_file: drive pad 0 from a script, to test
    // the renderer through the menus without a controller.
    rexglue_script_input::Install(config);
#endif
  }
#endif

  void OnPreLaunchModule() override {
    simpsons::LogLoadedImage(runtime()->memory()->virtual_membase());
#if SIMPSONS_NATIVE_RENDERER && !REX_PLATFORM_SWITCH
    screenshots_.Start(runtime()->graphics_system());
#endif
  }

#if SIMPSONS_NATIVE_RENDERER && !REX_PLATFORM_SWITCH
  void OnShutdown() override { screenshots_.Stop(); }
#endif

  // Override virtual hooks for customization:
  // void OnPostInitLogging() override {}
  // void OnPreSetup(rex::RuntimeConfig& config) override {}
  // void OnLoadXexImage(std::string& xex_image) override {}
  // void OnPostSetup() override {}
  // void OnCreateDialogs(rex::ui::ImGuiDrawer* drawer) override {}
  // void OnShutdown() override {}
  // void OnConfigurePaths(rex::PathConfig& paths) override {}

#if SIMPSONS_NATIVE_RENDERER && !REX_PLATFORM_SWITCH
 private:
  simpsons::dev::Screenshots screenshots_;
#endif
};
