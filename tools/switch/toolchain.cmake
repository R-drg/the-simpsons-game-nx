# Cross compile the static ReXGlue runtime with devkitA64 and libnx.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(REXGLUE_PLATFORM_SWITCH ON CACHE BOOL "Horizon target" FORCE)
set(REXGLUE_USE_VULKAN ON CACHE BOOL "Vulkan renderer" FORCE)
set(REXGLUE_USE_D3D12 OFF CACHE BOOL "D3D12 renderer" FORCE)
set(REXGLUE_ENABLE_TRACY OFF CACHE BOOL "Tracy profiler" FORCE)
set(REXGLUE_ENABLE_PERF_COUNTERS ON CACHE BOOL "GPU counters" FORCE)

if(DEFINED ENV{DEVKITPRO})
    set(_switch_dkp "$ENV{DEVKITPRO}")
else()
    set(_switch_dkp "/opt/devkitpro")
endif()
set(DEVKITPRO "${_switch_dkp}" CACHE PATH "devkitPro installation")
set(_switch_prefix "${DEVKITPRO}/devkitA64/bin/aarch64-none-elf-")
set(CMAKE_C_COMPILER "${_switch_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_switch_prefix}g++")
set(CMAKE_ASM_COMPILER "${_switch_prefix}gcc")
set(CMAKE_AR "${_switch_prefix}gcc-ar" CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${_switch_prefix}gcc-ranlib" CACHE FILEPATH "" FORCE)
set(CMAKE_NM "${_switch_prefix}nm" CACHE FILEPATH "" FORCE)
set(CMAKE_OBJCOPY "${_switch_prefix}objcopy" CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP "${_switch_prefix}strip" CACHE FILEPATH "" FORCE)
set(CMAKE_C_COMPILER_AR "${CMAKE_AR}")
set(CMAKE_C_COMPILER_RANLIB "${CMAKE_RANLIB}")
set(CMAKE_CXX_COMPILER_AR "${CMAKE_AR}")
set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_RANLIB}")

set(_switch_cpu "-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft")
set(_switch_defs "-D__SWITCH__ -DNX -D_GNU_SOURCE -DSPDLOG_NO_TZ_OFFSET")
set(_switch_common "${_switch_cpu} ${_switch_defs} -fPIE -ftls-model=local-exec")
set(CMAKE_C_FLAGS_INIT "${_switch_common} -ffunction-sections -fdata-sections -Wno-psabi")
set(CMAKE_CXX_FLAGS_INIT "${_switch_common} -ffunction-sections -fdata-sections -Wno-psabi")
set(CMAKE_ASM_FLAGS_INIT "${_switch_common}")

# Resolve libnx's linker-script path at configure time for reproducible builds.
set(_switch_specs "${DEVKITPRO}/libnx/switch.specs")
if(NOT EXISTS "${_switch_specs}")
    message(FATAL_ERROR "Missing libnx specs: ${_switch_specs}")
endif()
file(READ "${_switch_specs}" _switch_specs_text)
string(REPLACE "%:getenv(DEVKITPRO /libnx/switch.ld)"
    "${DEVKITPRO}/libnx/switch.ld" _switch_specs_text "${_switch_specs_text}")
file(WRITE "${CMAKE_BINARY_DIR}/switch.specs" "${_switch_specs_text}")
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-specs=${CMAKE_BINARY_DIR}/switch.specs -Wl,--gc-sections -Wl,--allow-multiple-definition -Wl,--no-relax")
set(CMAKE_DL_LIBS "")

include_directories(SYSTEM
    "${DEVKITPRO}/devkitA64/include"
    "${DEVKITPRO}/libnx/include"
    "${DEVKITPRO}/portlibs/switch/include")
link_directories(
    "${DEVKITPRO}/devkitA64/lib"
    "${DEVKITPRO}/libnx/lib"
    "${DEVKITPRO}/portlibs/switch/lib")
set(CMAKE_FIND_ROOT_PATH
    "${DEVKITPRO}/devkitA64"
    "${DEVKITPRO}/libnx"
    "${DEVKITPRO}/portlibs/switch")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
