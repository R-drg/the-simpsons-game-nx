#version 450
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copies a rectangle of a depth buffer into the R32_SFLOAT image the
// translated shaders sample a depth resolve (k_24_8, k_24_8_FLOAT) as, in one
// pass. Reading the depth through a buffer with two copies cost about 1 ms of
// GPU time per full-screen resolve on this GPU; this is a small fraction.
//
// Rebuild depth_to_float_cs.h after editing:
//   glslangValidator -S comp -V --target-env vulkan1.1
//     --vn kDepthToFloatCs -o depth_to_float_cs.h depth_to_float.cs.glsl
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform texture2D depth_buffer;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D dest;
layout(push_constant) uniform Params {
  ivec2 origin;
  uvec2 size;
};
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= size.x || p.y >= size.y) {
    return;
  }
  ivec2 c = origin + ivec2(p);
  imageStore(dest, c, vec4(texelFetch(depth_buffer, c, 0).r));
}
