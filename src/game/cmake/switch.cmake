# Horizon support adapted from project8-sw; code generation stays on the host.
# REXSDK_DIR is the ReXGlue SDK of TheSimpsonsGameRecomp with patches/rexglue-sdk
# applied (docs/building.md).
set(REXSDK_DIR "" CACHE PATH "Switch SDK source (patched)")
if(NOT EXISTS "${REXSDK_DIR}/cmake/rexglue_switch.cmake")
    message(FATAL_ERROR "Set REXSDK_DIR to the patched SDK (see docs/building.md)")
endif()
set(SIMPSONS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated/default" CACHE PATH "Generated game sources")
if(NOT EXISTS "${SIMPSONS_GENERATED_DIR}/sources.cmake")
    message(FATAL_ERROR "Generate the Simpsons translation units before cross-compiling")
endif()
add_subdirectory("${REXSDK_DIR}" rexglue-sdk)
include("${SIMPSONS_GENERATED_DIR}/sources.cmake")
add_library(simpsons_recomp OBJECT ${GENERATED_SOURCES})
target_include_directories(simpsons_recomp PRIVATE "${SIMPSONS_GENERATED_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/compat_switch")
target_link_libraries(simpsons_recomp PRIVATE rex::runtime)
target_compile_options(simpsons_recomp PRIVATE -O3 -ffp-contract=off
    -fno-tree-slp-vectorize -fno-tree-vectorize)

# The PC port's hand-written replacements for the game's costliest shaders
# (its full-screen post effects), compiled and embedded into the NRO: on the
# Switch those effects were half of the GPU's time per frame.
set(SIMPSONS_NATIVE_SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/native-renderer/native_shaders")
file(GLOB SIMPSONS_NATIVE_SHADER_FILES "${SIMPSONS_NATIVE_SHADER_SOURCES}/*.frag")
set(SIMPSONS_NATIVE_SHADER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/native_shaders")
set(SIMPSONS_NATIVE_SHADER_CPP "${CMAKE_CURRENT_BINARY_DIR}/native_shaders_embedded.cpp")
add_custom_command(
    OUTPUT "${SIMPSONS_NATIVE_SHADER_CPP}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${SIMPSONS_NATIVE_SHADER_SPV_DIR}"
    COMMAND python3 "${SIMPSONS_NATIVE_SHADER_SOURCES}/build_native_shaders.py"
            "${SIMPSONS_NATIVE_SHADER_SPV_DIR}" 0000400000000001
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_native_shaders.py"
            "${SIMPSONS_NATIVE_SHADER_SPV_DIR}" "${SIMPSONS_NATIVE_SHADER_CPP}"
    DEPENDS ${SIMPSONS_NATIVE_SHADER_FILES}
            "${SIMPSONS_NATIVE_SHADER_SOURCES}/build_native_shaders.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_native_shaders.py"
    COMMENT "Building the native replacement shaders"
    VERBATIM)

add_library(simpsons_host OBJECT src/main.cpp "${SIMPSONS_NATIVE_SHADER_CPP}" src/draw_telemetry.cpp
    src/frame_pacing.cpp src/ring_wait.cpp src/subtitles.cpp src/startup_check.cpp
    src/movie_decoder.cpp
    src/native/native_system.cpp src/native/native_renderer.cpp
    src/native/native_textures.cpp src/native/native_geometry_shaders.cpp
    "${REXSDK_DIR}/src/ui/rex_app.cpp"
    "${REXSDK_DIR}/src/ui/windowed_app_main_switch.cpp")
# The native renderer uses the SDK's shader translator and gamma shader, whose
# headers rexruntime does not export: it links graphics, glslang and xxHash
# privately.
target_include_directories(simpsons_host PRIVATE "${SIMPSONS_GENERATED_DIR}"
    "${REXSDK_DIR}/thirdparty/imgui"
    "${REXSDK_DIR}/src/graphics"
    "${REXSDK_DIR}/thirdparty/glslang"
    "${REXSDK_DIR}/thirdparty/xxHash")
target_link_libraries(simpsons_host PRIVATE rex::runtime)
# The SDK's FFmpeg, with the VP6 decoder of patches/rexglue-sdk/0002, decodes
# the cutscenes (src/movie_decoder.cpp).
target_link_libraries(simpsons_host PRIVATE libavcodec)
target_compile_definitions(simpsons_host PRIVATE REXGLUE_BUILD_CONFIG="$<CONFIG>")

add_custom_target(simpsons_switch_compile DEPENDS simpsons_recomp simpsons_host
    rexruntime rexglue_switch_startup)
if(NOT REXGLUE_SWITCH_COMPILE_ONLY)
    add_executable(simpsons $<TARGET_OBJECTS:simpsons_host> $<TARGET_OBJECTS:simpsons_recomp>)
    rexglue_switch_configure_target(simpsons NO_ENTRY)
    # The objects of simpsons_host do not carry its link libraries.
    target_link_libraries(simpsons PRIVATE libavcodec)
    rexglue_switch_add_nro(simpsons NAME "The Simpsons Game" AUTHOR "Rodrigo Ferreira" VERSION "1.0.1"
        ICON "${CMAKE_CURRENT_SOURCE_DIR}/switch/icon.jpg")
endif()
