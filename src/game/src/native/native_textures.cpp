// SPDX-License-Identifier: GPL-3.0-or-later
//
// Textures for the native renderer. See native_textures.h for the design.
//
// Every image here is kept in VK_IMAGE_LAYOUT_GENERAL once written, so the
// descriptors never need to know where an image came from. On this GPU the
// layout has no cost.

#include "native_textures.h"
#include "native_hash.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

#if defined(__SWITCH__)
extern "C" {
#include <switch/kernel/svc.h>
}
#endif

#include <fmt/format.h>
#include <xxhash.h>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory/utils.h>
#include <rex/graphics/pipeline/texture/conversion.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/util.h>

namespace simpsons::native {

namespace xenos = rex::graphics::xenos;
namespace texture_util = rex::graphics::texture_util;
namespace texture_conversion = rex::graphics::texture_conversion;
using rex::graphics::FormatInfo;
using rex::ui::vulkan::VulkanDevice;

namespace {

// How guest texels become host texels.
enum class Conversion : uint8_t {
  kUnsupported,
  // Byte order swap only.
  kCopy,
  // 16-bit packed with red in the low bits: red and blue fields swapped into
  // the Vulkan packed layout.
  kSwapRB565,
  kSwapRB1555,
  // RGBA4 with red in the low bits into Vulkan B4G4R4A4 (blue in the high bits).
  kRGBA4ToBGRA4,
  // Unsigned 10:11:11 and 11:11:10 (red first) into RGBA16 UNORM.
  kR10G11B11ToRGBA16,
  kR11G11B10ToRGBA16,
  // 4x4 blocks decompressed into R8G8 texels.
  kCTX1,
  // 4x4 4-bit alpha blocks into BC2 blocks with black color.
  kDXT3A,
};

struct HostFormat {
  VkFormat unorm = VK_FORMAT_UNDEFINED;
  VkFormat snorm = VK_FORMAT_UNDEFINED;
  Conversion conversion = Conversion::kUnsupported;
  // The host components for the guest components XYZW, in the xenos swizzle
  // encoding. From ReXGlue's VulkanTextureCache::kBestHostFormats.
  uint32_t swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  // The host image is block-compressed with the same 4x4 blocks as the guest.
  bool compressed = false;
  // Host bytes per guest block for the uncompressed conversions that change
  // the size (0 = same as the guest).
  uint32_t host_bytes_per_texel = 0;
};

constexpr uint32_t kSwizzleRRRR = xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR;
constexpr uint32_t kSwizzleRGGG = xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG;
constexpr uint32_t kSwizzleRGBB = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB;
constexpr uint32_t kSwizzleRGBA = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;

HostFormat GetHostFormat(xenos::TextureFormat format) {
  using F = xenos::TextureFormat;
  HostFormat h;
  switch (format) {
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B:
      h = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, Conversion::kCopy, kSwizzleRRRR};
      break;
    case F::k_1_5_5_5:
      h = {VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_UNDEFINED, Conversion::kSwapRB1555,
           kSwizzleRGBA};
      break;
    case F::k_5_6_5:
      h = {VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED, Conversion::kSwapRB565,
           kSwizzleRGBB};
      break;
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16:
      h = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, Conversion::kCopy, kSwizzleRGBA};
      break;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      h = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_UNDEFINED, Conversion::kCopy,
           kSwizzleRGBA};
      break;
    case F::k_8_8:
      h = {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, Conversion::kCopy, kSwizzleRGGG};
      break;
    case F::k_4_4_4_4:
      h = {VK_FORMAT_B4G4R4A4_UNORM_PACK16, VK_FORMAT_UNDEFINED, Conversion::kRGBA4ToBGRA4,
           kSwizzleRGBA};
      break;
    case F::k_10_11_11:
    case F::k_10_11_11_AS_16_16_16_16:
      h = {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_UNDEFINED, Conversion::kR10G11B11ToRGBA16,
           kSwizzleRGBB, false, 8};
      break;
    case F::k_11_11_10:
    case F::k_11_11_10_AS_16_16_16_16:
      h = {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_UNDEFINED, Conversion::kR11G11B10ToRGBA16,
           kSwizzleRGBB, false, 8};
      break;
    case F::k_DXT1:
    case F::k_DXT1_AS_16_16_16_16:
      h = {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGBA,
           true};
      break;
    case F::k_DXT2_3:
    case F::k_DXT2_3_AS_16_16_16_16:
      h = {VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGBA, true};
      break;
    case F::k_DXT4_5:
    case F::k_DXT4_5_AS_16_16_16_16:
      h = {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGBA, true};
      break;
    case F::k_DXN:
      h = {VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_SNORM_BLOCK, Conversion::kCopy, kSwizzleRGGG,
           true};
      break;
    case F::k_DXT5A:
      h = {VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK, Conversion::kCopy, kSwizzleRRRR,
           true};
      break;
    case F::k_DXT3A:
      // The alpha of a BC2 block: all four components read it.
      h = {VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_UNDEFINED, Conversion::kDXT3A,
           XE_GPU_MAKE_TEXTURE_SWIZZLE(A, A, A, A), true};
      break;
    case F::k_CTX1:
      h = {VK_FORMAT_R8G8_UNORM, VK_FORMAT_UNDEFINED, Conversion::kCTX1, kSwizzleRGGG, false, 2};
      break;
    case F::k_16:
      h = {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, Conversion::kCopy, kSwizzleRRRR};
      break;
    case F::k_16_16:
      h = {VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, Conversion::kCopy, kSwizzleRGGG};
      break;
    case F::k_16_16_16_16:
      h = {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, Conversion::kCopy,
           kSwizzleRGBA};
      break;
    case F::k_16_EXPAND:
    case F::k_16_FLOAT:
      h = {VK_FORMAT_R16_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRRRR};
      break;
    case F::k_16_16_EXPAND:
    case F::k_16_16_FLOAT:
      h = {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGGG};
      break;
    case F::k_16_16_16_16_EXPAND:
    case F::k_16_16_16_16_FLOAT:
      h = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGBA};
      break;
    case F::k_32_FLOAT:
      h = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRRRR};
      break;
    case F::k_32_32_FLOAT:
      h = {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGGG};
      break;
    case F::k_32_32_32_32_FLOAT:
      h = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_UNDEFINED, Conversion::kCopy, kSwizzleRGBA};
      break;
    default:
      break;
  }
  return h;
}

VkComponentSwizzle ComponentSwizzle(uint32_t swizzle, uint32_t component) {
  switch ((swizzle >> (3 * component)) & 0b111) {
    case 0:
      return VK_COMPONENT_SWIZZLE_R;
    case 1:
      return VK_COMPONENT_SWIZZLE_G;
    case 2:
      return VK_COMPONENT_SWIZZLE_B;
    case 3:
      return VK_COMPONENT_SWIZZLE_A;
    case 5:
      return VK_COMPONENT_SWIZZLE_ONE;
    default:
      return VK_COMPONENT_SWIZZLE_ZERO;
  }
}

// TextureCache::GuestToHostSwizzle in ReXGlue.
uint32_t GuestToHostSwizzle(uint32_t guest_swizzle, uint32_t host_format_swizzle) {
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t guest_component = (guest_swizzle >> (3 * i)) & 0b111;
    uint32_t host_component;
    if (guest_component >= xenos::XE_GPU_TEXTURE_SWIZZLE_0) {
      host_component = guest_component & 0b101;
    } else {
      host_component = (host_format_swizzle >> (3 * guest_component)) & 0b111;
    }
    host_swizzle |= host_component << (3 * i);
  }
  return host_swizzle;
}

// The part of a fetch constant that decides the texture's layout, hashed with
// the guest data to identify an image.
struct TextureLayoutKey {
  uint32_t base_page;
  uint32_t mip_page;
  uint32_t width_minus_1;
  uint32_t height_minus_1;
  uint32_t depth_or_array_size_minus_1;
  uint32_t pitch;
  uint32_t mip_max_level;
  uint32_t format;
  uint32_t dimension;
  uint32_t endianness;
  uint32_t tiled;
  uint32_t packed_mips;
};

// The byte at logical address a of swapped guest memory is at physical
// address a ^ SwapXor(endian): the swaps exchange bytes within aligned 16- or
// 32-bit words.
inline uint32_t SwapXor(xenos::Endian endian) {
  switch (endian) {
    case xenos::Endian::k8in16:
      return 1;
    case xenos::Endian::k8in32:
      return 3;
    case xenos::Endian::k16in32:
      return 2;
    default:
      return 0;
  }
}

void ConvertBlock(Conversion conversion, xenos::Endian endian, uint8_t* out, const uint8_t* in,
                  uint32_t guest_bytes, uint32_t out_row_pitch) {
  switch (conversion) {
    case Conversion::kCopy:
      if (guest_bytes < 4 && endian != xenos::Endian::kNone) {
        // Smaller than the swap unit: CopySwapBlock swaps whole units and
        // copied nothing, leaving these texels as whatever the staging
        // memory held. Gather each byte from where the swap puts it; `in`
        // is at a guest_bytes-aligned offset from the 4 KB-aligned base.
        const uintptr_t x = SwapXor(endian);
        for (uint32_t i = 0; i < guest_bytes; ++i) {
          out[i] = *reinterpret_cast<const uint8_t*>((reinterpret_cast<uintptr_t>(in) + i) ^ x);
        }
      } else {
        texture_conversion::CopySwapBlock(endian, out, in, guest_bytes);
      }
      break;
    case Conversion::kSwapRB565: {
      uint16_t v;
      texture_conversion::CopySwapBlock(endian, &v, in, 2);
      v = uint16_t(((v & 0x1F) << 11) | (v & 0x7E0) | (v >> 11));
      std::memcpy(out, &v, 2);
    } break;
    case Conversion::kSwapRB1555: {
      uint16_t v;
      texture_conversion::CopySwapBlock(endian, &v, in, 2);
      v = uint16_t((v & 0x83E0) | ((v & 0x1F) << 10) | ((v >> 10) & 0x1F));
      std::memcpy(out, &v, 2);
    } break;
    case Conversion::kRGBA4ToBGRA4: {
      uint16_t v;
      texture_conversion::CopySwapBlock(endian, &v, in, 2);
      uint32_t r = v & 0xF, g = (v >> 4) & 0xF, b = (v >> 8) & 0xF, a = v >> 12;
      v = uint16_t((b << 12) | (g << 8) | (r << 4) | a);
      std::memcpy(out, &v, 2);
    } break;
    case Conversion::kR10G11B11ToRGBA16:
    case Conversion::kR11G11B10ToRGBA16: {
      uint32_t v;
      texture_conversion::CopySwapBlock(endian, &v, in, 4);
      uint32_t r, g, b;
      uint16_t o[4];
      if (conversion == Conversion::kR10G11B11ToRGBA16) {
        r = v & 0x3FF;
        g = (v >> 10) & 0x7FF;
        b = v >> 21;
        o[0] = uint16_t(r * 65535 / 0x3FF);
        o[1] = uint16_t(g * 65535 / 0x7FF);
        o[2] = uint16_t(b * 65535 / 0x7FF);
      } else {
        r = v & 0x7FF;
        g = (v >> 11) & 0x7FF;
        b = v >> 22;
        o[0] = uint16_t(r * 65535 / 0x7FF);
        o[1] = uint16_t(g * 65535 / 0x7FF);
        o[2] = uint16_t(b * 65535 / 0x3FF);
      }
      o[3] = 0xFFFF;
      std::memcpy(out, o, 8);
    } break;
    case Conversion::kCTX1:
      texture_conversion::ConvertTexelCTX1ToR8G8(endian, out, in, out_row_pitch);
      break;
    case Conversion::kDXT3A:
      texture_conversion::ConvertTexelDXT3AToDXT3(endian, out, in, 16);
      break;
    default:
      break;
  }
}

}  // namespace

// Host layout of one level: every layer tightly packed, rows in whole blocks.
struct LevelLayout {
  uint32_t width, height;       // texels
  uint32_t blocks_x, blocks_y;  // guest blocks
  uint32_t host_row_bytes;      // one row of host blocks or texels
  uint32_t host_rows;           // rows of host blocks or texels
  VkDeviceSize offset;          // of layer 0
  VkDeviceSize layer_bytes;
};

// Everything the untiling of one texture needs, so it can run on a worker.
struct UntileJob {
  LevelLayout levels[xenos::kTextureMaxMips];
  uint32_t level_count, layers, width, height;
  texture_util::TextureGuestLayout layout;
  const uint8_t* base_data;
  const uint8_t* mip_data;
  uint8_t* staging;
  bool tiled, host_blocks;
  xenos::Endian endianness;
  xenos::TextureFormat format;
  Conversion conversion;
  uint32_t block_width, block_height, guest_bytes_per_block, bpb_log2, host_texel_bytes;
};

void RunUntile(const UntileJob& j) {
    const auto* levels = j.levels;
    const uint32_t level_count = j.level_count, layers = j.layers, width = j.width,
                   height = j.height;
    const texture_util::TextureGuestLayout& layout = j.layout;
    const uint8_t* base_data = j.base_data;
    const uint8_t* mip_data = j.mip_data;
    uint8_t* staging = j.staging;
    const bool host_blocks = j.host_blocks;
    const xenos::TextureFormat format = j.format;
    const uint32_t block_width = j.block_width, block_height = j.block_height,
                   guest_bytes_per_block = j.guest_bytes_per_block, bpb_log2 = j.bpb_log2,
                   host_texel_bytes = j.host_texel_bytes;
    for (uint32_t level = 0; level < level_count; ++level) {
      const LevelLayout& l = levels[level];
      // Which storage holds this level, and where in it.
      const uint8_t* level_source;
      const texture_util::TextureGuestLayout::Level* guest_level;
      bool is_packed = layout.packed_level != UINT32_MAX && level >= layout.packed_level;
      if (level == 0) {
        level_source = base_data;
        guest_level = &layout.base;
      } else {
        uint32_t stored = is_packed ? layout.packed_level : level;
        level_source = mip_data ? mip_data + layout.mip_offsets_bytes[stored] : nullptr;
        guest_level = &layout.mips[stored];
      }
      uint32_t offset_x_blocks = 0, offset_y_blocks = 0, offset_z = 0;
      if (is_packed) {
        texture_util::GetPackedMipOffset(width, height, 1, format, level, offset_x_blocks,
                                         offset_y_blocks, offset_z);
      }
      uint32_t row_pitch_bytes = guest_level->row_pitch_bytes;
      uint32_t pitch_blocks = row_pitch_bytes >> bpb_log2;
      // Tiled memory keeps aligned runs of blocks contiguous: 16 bytes, or 8
      // blocks of one byte. A plain copy moves a whole run per tiled-offset
      // computation instead of one block, 4 times fewer for 32-bit texels,
      // which was most of the time untiling a texture that changes every
      // frame. A run is 8 or 16 bytes at an offset aligned to its size, so it
      // is made of whole swap units and swapping it is swapping the memory
      // it came from, whatever the size of a texel.
      const uint32_t run_blocks = bpb_log2 == 0 ? 8 : (16u >> std::min(bpb_log2, 4u));
      const bool run_copy =
          j.tiled && j.conversion == Conversion::kCopy && run_blocks > 1 &&
          guest_bytes_per_block == (1u << bpb_log2) &&
          (host_blocks || block_width * host_texel_bytes == guest_bytes_per_block);
      for (uint32_t layer = 0; layer < layers; ++layer) {
        uint8_t* out = staging + l.offset + l.layer_bytes * layer;
        if (!level_source) {
          std::memset(out, 0, size_t(l.layer_bytes));
          continue;
        }
        const uint8_t* src = level_source + size_t(guest_level->array_slice_stride_bytes) * layer;
        for (uint32_t by = 0; by < l.blocks_y; ++by) {
          uint32_t gy = by + offset_y_blocks;
          uint8_t* out_row = out + size_t(by) * (host_blocks ? l.host_row_bytes
                                                             : l.host_row_bytes * block_height);
          // Linear: the row is contiguous, one copy (and swap) does it. The
          // video planes of the cutscenes (1280x720 and two 640x360, 8-bit,
          // every frame) took 15 ms a frame one texel at a time.
          if (!j.tiled && j.conversion == Conversion::kCopy &&
              (host_blocks || block_width * host_texel_bytes == guest_bytes_per_block)) {
            const size_t row_bytes = size_t(l.blocks_x) * guest_bytes_per_block;
            const uint8_t* src_row = src + size_t(gy) * row_pitch_bytes +
                                     size_t(offset_x_blocks) * guest_bytes_per_block;
            const uintptr_t start = reinterpret_cast<uintptr_t>(src_row);
            if (j.endianness == xenos::Endian::kNone ||
                (!(start & 3) && !(row_bytes & 3))) {
              texture_conversion::CopySwapBlock(j.endianness, out_row, src_row, row_bytes);
              continue;
            }
          }
          for (uint32_t bx = 0; bx < l.blocks_x; ++bx) {
            uint32_t gx = bx + offset_x_blocks;
            if (run_copy && !(gx & (run_blocks - 1)) && bx + run_blocks <= l.blocks_x) {
              size_t run_offset = size_t(texture_util::GetTiledOffset2D(
                  int32_t(gx), int32_t(gy), pitch_blocks, bpb_log2));
              texture_conversion::CopySwapBlock(j.endianness,
                                                out_row + size_t(bx) * guest_bytes_per_block,
                                                src + run_offset,
                                                run_blocks * guest_bytes_per_block);
              bx += run_blocks - 1;
              continue;
            }
            size_t src_offset;
            if (j.tiled) {
              src_offset = size_t(texture_util::GetTiledOffset2D(int32_t(gx), int32_t(gy),
                                                                 pitch_blocks, bpb_log2));
            } else {
              src_offset = size_t(gy) * row_pitch_bytes + size_t(gx) * guest_bytes_per_block;
            }
            uint8_t* out_block;
            if (host_blocks) {
              out_block = out_row + size_t(bx) * guest_bytes_per_block;
            } else {
              out_block = out_row + size_t(bx) * block_width * host_texel_bytes;
            }
            ConvertBlock(j.conversion, j.endianness, out_block, src + src_offset,
                         guest_bytes_per_block, l.host_row_bytes);
          }
        }
      }
    }
}

// A texture created off the renderer thread: the image, and staging memory of
// its own with the untiled data, which a later submission copies in.
struct StreamJob {
  UntileJob untile;
  VkImageCreateInfo info = {};
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer staging_buffer = VK_NULL_HANDLE;
  VkDeviceMemory staging_memory = VK_NULL_HANDLE;
  VkDeviceSize staging_size = 0;
  std::atomic<bool> done{false};
  bool ok = false;
};

REXCVAR_DEFINE_UINT32(nx_max_anisotropy, 2, "Switch",
                      "Most anisotropic filtering any texture gets (1 turns it off); the "
                      "game asks for it on floors, where it costs the GPU the most");

// New textures made on the spot each frame, up to this many: drawn the first
// time, not a frame or more later. Beyond it they stream in, so a new screen
// loading hundreds at once does not stall a frame on their allocations (each
// a kernel round trip). Streaming everything flickered: a texture missing for
// a frame shows as a flash of the empty texture on characters.
// Kept small: six of up to 512 KB each made a level load (a thousand new
// textures) cost 37 ms of renderer time per frame.
constexpr uint32_t kSyncCreatesPerFrame = 2;
constexpr VkDeviceSize kSyncCreateMaxBytes = 32 * 1024;

// New textures with at least this much host data stream in asynchronously.
// Nearly all of them: the cost that hitched was not the data but creating the
// image (memory allocation and GPU mapping are each a kernel round trip), and
// that is the same for a small texture as for a big one.
// Every one beyond the per-frame budget, however small: under 1 KB they were
// all made on the spot, and a level load made 116 in one frame (180 ms).
constexpr VkDeviceSize kStreamBytes = 0;

struct TextureCache::Texture {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFormat format_unorm = VK_FORMAT_UNDEFINED;
  VkFormat format_snorm = VK_FORMAT_UNDEFINED;
  xenos::DataDimension dimension = xenos::DataDimension::k2DOrStacked;
  uint32_t host_format_swizzle = kSwizzleRGBA;
  // Views by (host swizzle << 1 | signed).
  std::unordered_map<uint32_t, VkImageView> views;
  uint64_t last_used_frame = 0;
  // Submission that last sampled it: the image can be rewritten in place only
  // before the first use in a submission, since uploads run ahead of it.
  uint64_t last_used_submission = 0;
  uint64_t key_hash = 0;
  uint64_t content_hash = 0;
  // Still being created by the streaming worker.
  std::shared_ptr<StreamJob> stream;
  uint32_t level_count = 0;
  uint32_t layers = 0;
};

struct TextureCache::Impl {
  const VulkanDevice* device = nullptr;
  rex::memory::Memory* memory = nullptr;

  // Layout key hash -> texture. When the content changes, the image is
  // uploaded again in place: allocating a new image each time was a kernel
  // round trip (memory map) per changed texture, about 50 a second in play.
  std::unordered_map<uint64_t, std::unique_ptr<Texture>> textures;
  // Layout key hash -> content hash of the last check, per frame, so the data
  // of a texture used by many draws is hashed once per frame.
  struct Checked {
    uint64_t frame = UINT64_MAX;
    uint64_t texture_hash = 0;
    ChangeDetector base;
    ChangeDetector mips;
  };
  FlatMap64<Checked> checked;
  struct UploadStat {
    uint32_t address = 0, format = 0, width = 0, height = 0, count = 0;
    uint64_t bytes = 0, last_hash = 0;
  };
  std::unordered_map<uint64_t, UploadStat> upload_stats;
  void CountUpload(uint64_t key_hash, uint32_t address, uint32_t format, uint32_t width,
                   uint32_t height, uint64_t bytes, uint64_t content_hash) {
    if (upload_stats.size() > 4096) return;
    UploadStat& u = upload_stats[key_hash];
    u.address = address;
    u.format = format;
    u.width = width;
    u.height = height;
    ++u.count;
    u.bytes += bytes;
    u.last_hash = content_hash;
  }
  std::unordered_set<uint32_t> depth_words_wanted;

  std::unordered_map<uint32_t, std::unique_ptr<ResolveDestination>> resolve_destinations;

  // This frame's answers by the raw fetch constant, so a texture bound by many
  // draws is looked up once: the layout computation and the hashing are then
  // skipped for the rest of the frame.
  struct Memo {
    uint64_t frame;
    VkImageView view;
    Texture* texture;
    // Reused through this frame: the content checks would not look at the
    // texture's data before then (see ChangeDetector::TrustedUntil).
    uint64_t valid_until;
  };
  // Set by GetViewUncached for the memo.
  uint64_t last_valid_until = 0;
  // What the last GetView answered from (last_texture(), last_valid_until()).
  Texture* out_texture = nullptr;
  uint64_t out_valid_until = 0;
  uint32_t sync_creates_left = 0;
  FlatMap64<Memo> memo;
  uint64_t memo_frame = UINT64_MAX;
  // The texture GetViewUncached returned a view of (null for resolve
  // destinations and null views).
  Texture* last_texture = nullptr;

  std::unordered_map<uint32_t, VkSampler> samplers;

  VkImage null_images[3] = {};
  VkDeviceMemory null_memory[3] = {};
  VkImageView null_views[3] = {};  // 2D array, cube, 3D

  // Destroyed once the frame that last used them is complete.
  struct Retired {
    uint64_t frame;
    VkImage image;
    VkDeviceMemory memory;
    std::vector<VkImageView> views;
  };
  std::vector<Retired> retired;

  uint64_t frame = 0;
  uint64_t completed_frame = 0;
  float max_anisotropy = 1.0f;

  std::vector<uint8_t> scratch;

  // Upload worker: untiles into staging while the ring keeps recording.
  std::thread upload_thread;
  std::mutex upload_mutex;
  std::condition_variable upload_cv;
  std::condition_variable upload_done_cv;
  std::deque<std::function<void()>> upload_jobs;
  uint32_t upload_running = 0;
  bool upload_stop = false;

  void EnqueueUpload(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lock(upload_mutex);
      upload_jobs.push_back(std::move(job));
    }
    upload_cv.notify_one();
  }

  void UploadThread() {
#if defined(__SWITCH__)
    svcSetThreadPriority(CUR_THREAD_HANDLE, 0x3B);
#endif
    std::unique_lock<std::mutex> lock(upload_mutex);
    while (true) {
      upload_cv.wait(lock, [&] { return upload_stop || !upload_jobs.empty(); });
      if (upload_jobs.empty() && upload_stop) {
        return;
      }
      std::function<void()> job = std::move(upload_jobs.front());
      upload_jobs.pop_front();
      ++upload_running;
      lock.unlock();
      job();
      lock.lock();
      --upload_running;
      if (upload_jobs.empty() && !upload_running) {
        upload_done_cv.notify_all();
      }
    }
  }

  // Streaming worker: creates large new textures and fills their staging.
  // Two of them, just below the game's threads in priority. Above them, a
  // cutscene loading a thousand textures had both take cores from the game,
  // which stalled; one alone at the lowest priority could leave a new screen
  // drawn with the empty texture for seconds.
  std::thread stream_threads[2];
  std::mutex stream_mutex;
  std::condition_variable stream_cv;
  std::deque<std::shared_ptr<StreamJob>> stream_jobs;
  bool stream_stop = false;

  void StreamThread() {
#if defined(__SWITCH__)
    svcSetThreadPriority(CUR_THREAD_HANDLE, 0x3C);
#endif
    while (true) {
      std::shared_ptr<StreamJob> job;
      {
        std::unique_lock<std::mutex> lock(stream_mutex);
        stream_cv.wait(lock, [&] { return stream_stop || !stream_jobs.empty(); });
        if (stream_stop) return;
        job = stream_jobs.front();
        stream_jobs.pop_front();
      }
      job->ok = rex::ui::vulkan::util::CreateDedicatedAllocationImage(
          device, job->info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, job->image,
          job->memory);
      if (job->ok) {
        job->ok = rex::ui::vulkan::util::CreateDedicatedAllocationBuffer(
            device, job->staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            rex::ui::vulkan::util::MemoryPurpose::kUpload, job->staging_buffer,
            job->staging_memory);
      }
      void* mapping = nullptr;
      if (job->ok && device->functions().vkMapMemory(device->device(), job->staging_memory, 0,
                                                     VK_WHOLE_SIZE, 0, &mapping) != VK_SUCCESS) {
        job->ok = false;
      }
      if (job->ok) {
        job->untile.staging = static_cast<uint8_t*>(mapping);
        RunUntile(job->untile);
      }
      job->done.store(true, std::memory_order_release);
    }
  }

  void DestroyStreamJob(StreamJob& job) {
    const auto& dfn = device->functions();
    VkDevice d = device->device();
    if (job.staging_buffer) dfn.vkDestroyBuffer(d, job.staging_buffer, nullptr);
    if (job.staging_memory) dfn.vkFreeMemory(d, job.staging_memory, nullptr);
    job.staging_buffer = VK_NULL_HANDLE;
    job.staging_memory = VK_NULL_HANDLE;
  }

  struct RetiredStaging {
    uint64_t frame;
    std::shared_ptr<StreamJob> job;
  };
  std::vector<RetiredStaging> retired_staging;

  // The caller runs queued jobs itself rather than sleeping: it is waiting
  // for them anyway, and the worker shares its cores with the game's threads,
  // so it can sit preempted while the renderer waits on it.
  void WaitForUploads() {
    std::unique_lock<std::mutex> lock(upload_mutex);
    while (!upload_jobs.empty()) {
      std::function<void()> job = std::move(upload_jobs.front());
      upload_jobs.pop_front();
      ++upload_running;
      lock.unlock();
      job();
      lock.lock();
      --upload_running;
    }
    upload_done_cv.wait(lock, [&] { return upload_jobs.empty() && !upload_running; });
  }

  void Retire(VkImage image, VkDeviceMemory memory, std::vector<VkImageView> views,
              uint64_t last_used) {
    retired.push_back({last_used, image, memory, std::move(views)});
  }

  uint64_t views_epoch = 0;

  // Destruction of what the GPU no longer uses, on a thread of its own: each
  // freed image is a kernel round trip (unmapping and freeing), and doing them
  // at the frame boundary on the ring thread showed as bursts of lost time.
  std::thread reap_thread;
  std::mutex reap_mutex;
  std::condition_variable reap_cv;
  std::deque<std::function<void()>> reap_jobs;
  bool reap_stop = false;

  void ReapThread() {
#if defined(__SWITCH__)
    svcSetThreadPriority(CUR_THREAD_HANDLE, 0x3D);
#endif
    std::unique_lock<std::mutex> lock(reap_mutex);
    while (true) {
      reap_cv.wait(lock, [&] { return reap_stop || !reap_jobs.empty(); });
      if (reap_jobs.empty() && reap_stop) return;
      auto job = std::move(reap_jobs.front());
      reap_jobs.pop_front();
      lock.unlock();
      job();
      lock.lock();
    }
  }

  void Reap(std::function<void()> job) {
    if (!reap_thread.joinable()) {
      job();
      return;
    }
    {
      std::lock_guard<std::mutex> lock(reap_mutex);
      reap_jobs.push_back(std::move(job));
    }
    reap_cv.notify_one();
  }

  // Open addressing (std::unordered_map lookups were 4% of the ring thread);
  // a destroyed view's id is zeroed rather than erased, and a reused handle
  // then gets a new one.
  FlatMap64<uint64_t> view_ids;
  TextureCache::ComputeCopy compute_copy;
  std::unordered_map<uint32_t, bool> storage_capable;
  // Bumped whenever the memo is dropped: an answer for some fetch changed.
  uint64_t answers_epoch = 1;
  uint64_t next_view_id = 1;
  void ForgetViews(const std::vector<VkImageView>& views) {
    for (VkImageView v : views) {
      if (uint64_t* id = view_ids.find(uint64_t(reinterpret_cast<uintptr_t>(v)))) *id = 0;
    }
  }

  void DestroyLater(VkImage image, VkDeviceMemory memory, std::vector<VkImageView> views) {
    if (!views.empty()) {
      // Now, not when the reaper gets to it: before a handle can be reused.
      ++views_epoch;
      ForgetViews(views);
    }
    Reap([this, image, memory, views = std::move(views)]() {
      const auto& dfn = device->functions();
      VkDevice d = device->device();
      for (VkImageView v : views) dfn.vkDestroyImageView(d, v, nullptr);
      if (image) dfn.vkDestroyImage(d, image, nullptr);
      if (memory) dfn.vkFreeMemory(d, memory, nullptr);
    });
  }

  void DestroyNow(VkImage image, VkDeviceMemory memory, const std::vector<VkImageView>& views) {
    if (!views.empty()) {
      ++views_epoch;
      ForgetViews(views);
    }
    const auto& dfn = device->functions();
    VkDevice d = device->device();
    for (VkImageView v : views) {
      dfn.vkDestroyImageView(d, v, nullptr);
    }
    if (image) dfn.vkDestroyImage(d, image, nullptr);
    if (memory) dfn.vkFreeMemory(d, memory, nullptr);
  }
};

TextureCache::TextureCache() : impl_(std::make_unique<Impl>()) {}
TextureCache::~TextureCache() { Shutdown(); }

void TextureCache::WaitForUploads() { impl_->WaitForUploads(); }

bool TextureCache::Initialize(const VulkanDevice* device, rex::memory::Memory* memory) {
  Impl& im = *impl_;
  im.upload_thread = std::thread([&im]() { im.UploadThread(); });
  for (std::thread& thread : im.stream_threads) {
    thread = std::thread([&im]() { im.StreamThread(); });
  }
  im.reap_thread = std::thread([&im]() { im.ReapThread(); });
  im.device = device;
  im.memory = memory;
  im.max_anisotropy =
      device->properties().samplerAnisotropy
          ? std::min({16.0f, device->properties().maxSamplerAnisotropy,
                      float(std::max(REXCVAR_GET(nx_max_anisotropy), 1u))})
          : 1.0f;

  const auto& dfn = device->functions();
  VkDevice d = device->device();
  for (uint32_t i = 0; i < 3; ++i) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = i == 2 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    info.flags = i == 1 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {1, 1, 1};
    info.mipLevels = 1;
    info.arrayLayers = i == 1 ? 6 : 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!rex::ui::vulkan::util::CreateDedicatedAllocationImage(
            device, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, im.null_images[i],
            im.null_memory[i])) {
      REXLOG_ERROR("[native] Could not create the null textures");
      return false;
    }
    VkImageViewCreateInfo view = {};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = im.null_images[i];
    view.viewType = i == 0 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                           : (i == 1 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_3D);
    view.format = info.format;
    view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
    if (dfn.vkCreateImageView(d, &view, nullptr, &im.null_views[i]) != VK_SUCCESS) {
      return false;
    }
  }
  return true;
}

void TextureCache::InitializeNullImages(VkCommandBuffer commands) {
  Impl& im = *impl_;
  const auto& dfn = im.device->functions();
  for (uint32_t i = 0; i < 3; ++i) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = im.null_images[i];
    barrier.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
    dfn.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);
    VkClearColorValue black = {};
    VkImageSubresourceRange range = rex::ui::vulkan::util::InitializeSubresourceRange();
    dfn.vkCmdClearColorImage(commands, im.null_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &black, 1, &range);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dfn.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
  }
}

void TextureCache::Shutdown() {
  if (!impl_ || !impl_->device) {
    return;
  }
  Impl& im = *impl_;
  if (im.upload_thread.joinable()) {
    {
      std::lock_guard<std::mutex> lock(im.upload_mutex);
      im.upload_stop = true;
    }
    im.upload_cv.notify_all();
    im.upload_thread.join();
  }
  {
    {
      std::lock_guard<std::mutex> lock(im.stream_mutex);
      im.stream_stop = true;
    }
    im.stream_cv.notify_all();
    for (std::thread& thread : im.stream_threads) {
      if (thread.joinable()) thread.join();
    }
  }
  if (im.reap_thread.joinable()) {
    {
      std::lock_guard<std::mutex> lock(im.reap_mutex);
      im.reap_stop = true;
    }
    im.reap_cv.notify_all();
    im.reap_thread.join();  // runs what is still queued first
  }
  for (auto& retired : im.retired_staging) {
    im.DestroyStreamJob(*retired.job);
  }
  im.retired_staging.clear();
  const auto& dfn = im.device->functions();
  VkDevice d = im.device->device();
  for (auto& r : im.retired) {
    im.DestroyNow(r.image, r.memory, r.views);
  }
  im.retired.clear();
  for (auto& [hash, texture] : im.textures) {
    std::vector<VkImageView> views;
    for (auto& [k, v] : texture->views) views.push_back(v);
    if (texture->stream) {
      // Never finished: the job owns the image.
      im.DestroyStreamJob(*texture->stream);
      im.DestroyNow(texture->stream->image, texture->stream->memory, views);
      continue;
    }
    im.DestroyNow(texture->image, texture->memory, views);
  }
  im.textures.clear();
  for (auto& [address, dest] : im.resolve_destinations) {
    std::vector<VkImageView> views;
    if (dest->view) views.push_back(dest->view);
    for (auto& [k, v] : dest->texture_views) views.push_back(v);
    im.DestroyNow(dest->image, dest->memory, views);
    for (auto& sized : dest->sized) {
      std::vector<VkImageView> sized_views;
      for (auto& [k, v] : sized->views) sized_views.push_back(v);
      im.DestroyNow(sized->image, sized->memory, sized_views);
    }
  }
  im.resolve_destinations.clear();
  for (auto& [k, s] : im.samplers) {
    dfn.vkDestroySampler(d, s, nullptr);
  }
  im.samplers.clear();
  for (uint32_t i = 0; i < 3; ++i) {
    im.DestroyNow(im.null_images[i], im.null_memory[i],
                  im.null_views[i] ? std::vector<VkImageView>{im.null_views[i]}
                                   : std::vector<VkImageView>{});
    im.null_images[i] = VK_NULL_HANDLE;
    im.null_memory[i] = VK_NULL_HANDLE;
    im.null_views[i] = VK_NULL_HANDLE;
  }
  im.device = nullptr;
}

void TextureCache::BeginFrame(uint64_t frame, uint64_t completed_frame) {
  Impl& im = *impl_;
  im.frame = frame;
  im.completed_frame = completed_frame;
  im.sync_creates_left = kSyncCreatesPerFrame;
  // Destroy what the GPU no longer uses.
  auto it = std::remove_if(im.retired.begin(), im.retired.end(), [&](Impl::Retired& r) {
    if (r.frame > completed_frame) {
      return false;
    }
    im.DestroyLater(r.image, r.memory, std::move(r.views));
    return true;
  });
  im.retired.erase(it, im.retired.end());
  for (auto r = im.retired_staging.begin(); r != im.retired_staging.end();) {
    if (r->frame <= completed_frame) {
      std::shared_ptr<StreamJob> job = r->job;
      im.Reap([&im, job]() { im.DestroyStreamJob(*job); });
      r = im.retired_staging.erase(r);
    } else {
      ++r;
    }
  }
  // Evict textures unused for about 20 seconds; the old content of changing
  // textures (a video, a render-to-texture the CPU fills) is what this frees.
  constexpr uint64_t kEvictFrames = 600;
  if (frame % 60 == 0) {
    for (auto t = im.textures.begin(); t != im.textures.end();) {
      Texture& texture = *t->second;
      if (texture.stream) {
        ++t;
        continue;
      }
      if (texture.last_used_frame + kEvictFrames < frame) {
        std::vector<VkImageView> views;
        for (auto& [k, v] : texture.views) views.push_back(v);
        im.Retire(texture.image, texture.memory, std::move(views), texture.last_used_frame);
        t = im.textures.erase(t);
      } else {
        ++t;
      }
    }
    if (im.checked.size() > 16384) {
      im.checked.clear();
    }
  }
  stats_.live_textures = im.textures.size();
}

VkImageView TextureCache::GetResolveDestinationView2D(uint32_t address, uint32_t guest_swizzle,
                                                      uint32_t* width_out, uint32_t* height_out) {
  Impl& im = *impl_;
  auto it = im.resolve_destinations.find(address);
  if (it == im.resolve_destinations.end()) {
    return VK_NULL_HANDLE;
  }
  ResolveDestination& dest = *it->second;
  *width_out = dest.width;
  *height_out = dest.height;
  uint32_t swizzle = GuestToHostSwizzle(guest_swizzle, dest.host_swizzle);
  uint32_t key = swizzle | (1u << 31);
  auto v = dest.texture_views.find(key);
  if (v != dest.texture_views.end()) {
    return v->second;
  }
  VkImageViewCreateInfo view = {};
  view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view.image = dest.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = dest.format;
  view.components = {ComponentSwizzle(swizzle, 0), ComponentSwizzle(swizzle, 1),
                     ComponentSwizzle(swizzle, 2), ComponentSwizzle(swizzle, 3)};
  view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
  VkImageView result = VK_NULL_HANDLE;
  if (im.device->functions().vkCreateImageView(im.device->device(), &view, nullptr, &result) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  dest.texture_views.emplace(key, result);
  return result;
}

uint64_t TextureCache::views_epoch() const { return impl_->views_epoch; }

uint64_t TextureCache::ViewId(VkImageView view) {
  Impl& im = *impl_;
  if (im.view_ids.size() > 65536) {
    // Mostly destroyed views by now: start over. Ids already handed out stop
    // matching (ViewAlive), so whatever holds them rebuilds.
    im.view_ids.clear();
  }
  uint64_t& id = im.view_ids[uint64_t(reinterpret_cast<uintptr_t>(view))];
  if (!id) id = im.next_view_id++;
  return id;
}

bool TextureCache::ViewAlive(VkImageView view, uint64_t id) const {
  const uint64_t* known = impl_->view_ids.find(uint64_t(reinterpret_cast<uintptr_t>(view)));
  return known && *known == id;
}

uint64_t TextureCache::answers_epoch() const { return impl_->answers_epoch; }

void* TextureCache::last_texture() const { return impl_->out_texture; }

uint64_t TextureCache::last_valid_until() const { return impl_->out_valid_until; }

void TextureCache::Touch(void* texture, uint64_t submission) {
  auto* t = static_cast<Texture*>(texture);
  t->last_used_submission = submission;
  t->last_used_frame = impl_->frame;
}

std::string TextureCache::TakeUploadReport() {
  auto& stats = impl_->upload_stats;
  if (stats.empty()) return {};
  std::vector<const Impl::UploadStat*> sorted;
  for (auto& [k, u] : stats) sorted.push_back(&u);
  std::sort(sorted.begin(), sorted.end(),
            [](auto* a, auto* b) { return a->count > b->count; });
  std::string line;
  for (size_t i = 0; i < sorted.size() && i < 6; ++i) {
    const auto& u = *sorted[i];
    line += fmt::format("{}{:08X} fmt {} {}x{}: {} ({} KB, last {:04X})", i ? " | " : "",
                        u.address, u.format, u.width, u.height, u.count, u.bytes >> 10,
                        uint32_t(u.last_hash & 0xFFFF));
  }
  stats.clear();
  return line;
}

bool TextureCache::WantsDepthWords(uint32_t address) const {
  return impl_->depth_words_wanted.count(address) != 0;
}

void TextureCache::SetComputeCopy(ComputeCopy copy) { impl_->compute_copy = std::move(copy); }

bool TextureCache::StorageCapable(VkFormat format) {
  Impl& im = *impl_;
  auto it = im.storage_capable.find(uint32_t(format));
  if (it != im.storage_capable.end()) return it->second;
  VkFormatProperties properties = {};
  im.device->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
      im.device->physical_device(), format, &properties);
  bool capable = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
  im.storage_capable.emplace(uint32_t(format), capable);
  return capable;
}

void TextureCache::CopyToSized(ResolveDestination& dest, VkCommandBuffer commands) {
  if (dest.sized.empty()) {
    return;
  }
  const auto& dfn = impl_->device->functions();
  VkMemoryBarrier between = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
  dfn.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &between, 0, nullptr, 0,
                           nullptr);
  for (auto& sized : dest.sized) {
    if (sized->storage_view && dest.view && impl_->compute_copy &&
        impl_->compute_copy(commands, dest.view, sized->storage_view, 0, 0, 0, 0,
                            std::min(sized->width, dest.width),
                            std::min(sized->height, dest.height))) {
      sized->filled = true;
      continue;
    }
    VkImageCopy region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.extent = {std::min(sized->width, dest.width), std::min(sized->height, dest.height), 1};
    dfn.vkCmdCopyImage(commands, dest.image, VK_IMAGE_LAYOUT_GENERAL, sized->image,
                       VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    sized->filled = true;
  }
}

const ResolveDestination* TextureCache::FindResolveDestination(uint32_t address) const {
  auto it = impl_->resolve_destinations.find(address);
  return it == impl_->resolve_destinations.end() ? nullptr : it->second.get();
}

ResolveDestination* TextureCache::GetResolveDestination(uint32_t address, uint32_t width,
                                                        uint32_t height, VkFormat format,
                                                        uint32_t host_swizzle,
                                                        VkCommandBuffer commands) {
  Impl& im = *impl_;
  const auto& dfn = im.device->functions();
  VkDevice d = im.device->device();
  auto it = im.resolve_destinations.find(address);
  if (it != im.resolve_destinations.end()) {
    ResolveDestination& dest = *it->second;
    if (dest.width >= width && dest.height >= height && dest.format == format) {
      if (dest.host_swizzle != host_swizzle) {
        dest.host_swizzle = host_swizzle;
      }
      dest.last_written_frame = im.frame;
      return &dest;
    }
    width = std::max(width, dest.width);
    height = std::max(height, dest.height);
    std::vector<VkImageView> views;
    if (dest.view) views.push_back(dest.view);
    for (auto& [k, v] : dest.texture_views) views.push_back(v);
    im.Retire(dest.image, dest.memory, std::move(views), im.frame);
    for (auto& sized : dest.sized) {
      std::vector<VkImageView> sized_views;
      for (auto& [k, v] : sized->views) sized_views.push_back(v);
      im.Retire(sized->image, sized->memory, std::move(sized_views), im.frame);
    }
    im.resolve_destinations.erase(it);
    im.memo.clear();
    ++im.answers_epoch;
  }

  // Fetches of this address that were answered from memory must now find it.
  im.memo.clear();
  ++im.answers_epoch;
  auto dest = std::make_unique<ResolveDestination>();
  dest->address = address;
  dest->width = width;
  dest->height = height;
  dest->format = format;
  dest->host_swizzle = host_swizzle;
  dest->last_written_frame = im.frame;
  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  // Depth resolves, and color ones where the format allows, are written by
  // compute passes.
  if (format == VK_FORMAT_R32_SFLOAT || StorageCapable(format)) {
    info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!rex::ui::vulkan::util::CreateDedicatedAllocationImage(
          im.device, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, dest->image,
          dest->memory)) {
    REXLOG_ERROR("[native] Could not create a {}x{} resolve destination", width, height);
    return nullptr;
  }
  VkImageViewCreateInfo view = {};
  view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view.image = dest->image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
  dfn.vkCreateImageView(d, &view, nullptr, &dest->view);
  // Resolve destinations stay in GENERAL: they are copied into and sampled
  // alternately, and on this GPU the layout has no cost.
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = dest->image;
  barrier.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
  dfn.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  VkClearColorValue black = {};
  VkImageSubresourceRange range = rex::ui::vulkan::util::InitializeSubresourceRange();
  dfn.vkCmdClearColorImage(commands, dest->image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
  ResolveDestination* result = dest.get();
  im.resolve_destinations.emplace(address, std::move(dest));
  return result;
}

VkImageView TextureCache::GetView(const xenos::xe_gpu_texture_fetch_t& fetch,
                                  xenos::FetchOpDimension dimension, bool is_signed,
                                  UploadTarget& upload) {
  Impl& im = *impl_;
  im.out_texture = nullptr;
  im.out_valid_until = im.frame;
  VkImageView null_view = dimension == xenos::FetchOpDimension::kCube
                              ? im.null_views[1]
                              : (dimension == xenos::FetchOpDimension::k3DOrStacked &&
                                         fetch.dimension == xenos::DataDimension::k3D
                                     ? im.null_views[2]
                                     : im.null_views[0]);
  if (fetch.type != xenos::FetchConstantType::kTexture) {
    return null_view;
  }
  // Entries are overwritten, not cleared each frame: freeing and allocating
  // map nodes every frame contended the allocator lock with the pipeline
  // compiler thread.
  if (im.memo.size() > 16384) {
    im.memo.clear();
    ++im.answers_epoch;
  }
  uint64_t memo_key = XXH3_64bits_withSeed(&fetch, sizeof(fetch),
                                           uint64_t(dimension) | (uint64_t(is_signed) << 2));
  Impl::Memo& memo = im.memo[memo_key];
  if (memo.view && (memo.frame == im.frame || memo.valid_until >= im.frame)) {
    if (memo.texture) {
      memo.texture->last_used_submission = upload.submission;
      // What frees the texture once the GPU is done with it goes by this.
      memo.texture->last_used_frame = im.frame;
    }
    im.out_texture = memo.texture;
    im.out_valid_until = memo.texture ? memo.valid_until : im.frame;
    return memo.view;
  }
  im.last_valid_until = im.frame;
  VkImageView result_view = GetViewUncached(fetch, dimension, is_signed, upload, null_view);
  // GetViewUncached may have cleared the memo (a texture was replaced).
  im.memo[memo_key] = Impl::Memo{im.frame, result_view, im.last_texture,
                                 im.last_texture ? im.last_valid_until : im.frame};
  im.out_texture = im.last_texture;
  im.out_valid_until = im.last_texture ? im.last_valid_until : im.frame;
  return result_view;
}

VkImageView TextureCache::GetViewUncached(const xenos::xe_gpu_texture_fetch_t& fetch,
                                          xenos::FetchOpDimension dimension, bool is_signed,
                                          UploadTarget& upload, VkImageView null_view) {
  Impl& im = *impl_;
  const auto& dfn = im.device->functions();
  VkDevice d = im.device->device();
  im.last_texture = nullptr;

  uint32_t width_minus_1, height_minus_1, depth_minus_1, base_page, mip_page, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &width_minus_1, &height_minus_1,
                                                 &depth_minus_1, &base_page, &mip_page, nullptr,
                                                 &mip_max_level);
  if (!base_page && !mip_page) {
    return null_view;
  }

  // A texture a resolve wrote: the image, not the memory.
  if (fetch.dimension != xenos::DataDimension::k3D &&
      fetch.dimension != xenos::DataDimension::kCube &&
      dimension != xenos::FetchOpDimension::kCube) {
    auto it = im.resolve_destinations.find(base_page << 12);
    if (it != im.resolve_destinations.end() && it->second->format == VK_FORMAT_R32_SFLOAT) {
      // A depth resolve: floats for depth formats, the memory words for the rest.
      xenos::TextureFormat format = rex::graphics::GetBaseFormat(fetch.format);
      bool depth_format = format == xenos::TextureFormat::k_24_8 ||
                          format == xenos::TextureFormat::k_24_8_FLOAT;
      static uint32_t logged_formats = 0;
      if (!(logged_formats & (1u << (uint32_t(format) & 31)))) {
        logged_formats |= 1u << (uint32_t(format) & 31);
        REXLOG_INFO("[native] Depth resolve at {:08X} sampled as texture format {}",
                    base_page << 12, uint32_t(format));
      }
      if (!depth_format) {
        im.depth_words_wanted.insert(base_page << 12);
        auto words = im.resolve_destinations.find((base_page << 12) | kDepthWordsTag);
        if (words != im.resolve_destinations.end()) {
          it = words;
        }
      }
    }
    if (it != im.resolve_destinations.end()) {
      ResolveDestination& dest = *it->second;
      uint32_t swizzle = GuestToHostSwizzle(fetch.swizzle, dest.host_swizzle);
      const uint32_t fetch_width = width_minus_1 + 1, fetch_height = height_minus_1 + 1;
      if ((fetch_width != dest.width || fetch_height != dest.height) && fetch_width <= 8192 &&
          fetch_height <= 8192 && depth_minus_1 == 0) {
        ResolveDestination::Sized* sized = nullptr;
        for (auto& candidate : dest.sized) {
          if (candidate->width == fetch_width && candidate->height == fetch_height) {
            sized = candidate.get();
          }
        }
        if (!sized && dest.sized.size() < 4) {
          auto created = std::make_unique<ResolveDestination::Sized>();
          created->width = fetch_width;
          created->height = fetch_height;
          VkImageCreateInfo info = {};
          info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
          info.imageType = VK_IMAGE_TYPE_2D;
          info.format = dest.format;
          info.extent = {fetch_width, fetch_height, 1};
          info.mipLevels = 1;
          info.arrayLayers = 1;
          info.samples = VK_SAMPLE_COUNT_1_BIT;
          info.tiling = VK_IMAGE_TILING_OPTIMAL;
          info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
          const bool storage = im.compute_copy && StorageCapable(dest.format);
          if (storage) info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
          info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          if (rex::ui::vulkan::util::CreateDedicatedAllocationImage(
                  im.device, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal,
                  created->image, created->memory)) {
            if (storage) {
              VkImageViewCreateInfo plain = {};
              plain.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
              plain.image = created->image;
              plain.viewType = VK_IMAGE_VIEW_TYPE_2D;
              plain.format = dest.format;
              plain.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
              if (dfn.vkCreateImageView(d, &plain, nullptr, &created->storage_view) !=
                  VK_SUCCESS) {
                created->storage_view = VK_NULL_HANDLE;
              } else {
                created->views.emplace(0xFFFFFFFFu, created->storage_view);
              }
            }
            // Into GENERAL before this submission's draws; filled by the next
            // resolve to this address, which runs after them.
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = created->image;
            barrier.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
            dfn.vkCmdPipelineBarrier(upload.setup_commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT |
                                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &barrier);
            REXLOG_INFO("[native] Resolve at {:08X} ({}x{}) sampled as {}x{}: exact-size copy",
                        dest.address, dest.width, dest.height, fetch_width, fetch_height);
            sized = created.get();
            dest.sized.push_back(std::move(created));
          }
        }
        // Until the first resolve after it was made, the whole image (once).
        if (sized && sized->filled) {
          auto sv = sized->views.find(swizzle);
          if (sv != sized->views.end()) {
            ++stats_.resolved_hits;
            return sv->second;
          }
          VkImageViewCreateInfo view = {};
          view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
          view.image = sized->image;
          view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
          view.format = dest.format;
          view.components = {ComponentSwizzle(swizzle, 0), ComponentSwizzle(swizzle, 1),
                             ComponentSwizzle(swizzle, 2), ComponentSwizzle(swizzle, 3)};
          view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
          VkImageView result = VK_NULL_HANDLE;
          if (dfn.vkCreateImageView(d, &view, nullptr, &result) != VK_SUCCESS) {
            return null_view;
          }
          sized->views.emplace(swizzle, result);
          ++stats_.resolved_hits;
          return result;
        }
      }
      auto v = dest.texture_views.find(swizzle);
      if (v != dest.texture_views.end()) {
        ++stats_.resolved_hits;
        return v->second;
      }
      VkImageViewCreateInfo view = {};
      view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view.image = dest.image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      view.format = dest.format;
      view.components = {ComponentSwizzle(swizzle, 0), ComponentSwizzle(swizzle, 1),
                         ComponentSwizzle(swizzle, 2), ComponentSwizzle(swizzle, 3)};
      view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
      VkImageView result = VK_NULL_HANDLE;
      if (dfn.vkCreateImageView(d, &view, nullptr, &result) != VK_SUCCESS) {
        return null_view;
      }
      dest.texture_views.emplace(swizzle, result);
      ++stats_.resolved_hits;
      return result;
    }
  }

  xenos::TextureFormat format = rex::graphics::GetBaseFormat(fetch.format);
  HostFormat host_format = GetHostFormat(format);
  // 1D and 3D textures are not handled. 1D ones were tried as 2D textures a
  // texel tall; the first run with them had Free Skate's world geometry
  // stretched and untextured, which it never was with them drawn empty.
  if (host_format.conversion == Conversion::kUnsupported ||
      fetch.dimension == xenos::DataDimension::k3D ||
      fetch.dimension == xenos::DataDimension::k1D) {
    ++stats_.unsupported;
    ++stats_.pending;
    static std::unordered_set<uint32_t> logged;
    uint32_t what = (uint32_t(format) << 4) | uint32_t(fetch.dimension);
    if (logged.insert(what).second) {
      REXLOG_WARN("[native] Texture not handled: format {}, dimension {}, {}x{}x{}",
                  uint32_t(format), uint32_t(fetch.dimension), width_minus_1 + 1,
                  height_minus_1 + 1, depth_minus_1 + 1);
    }
    return null_view;
  }
  if ((fetch.dimension == xenos::DataDimension::kCube) !=
      (dimension == xenos::FetchOpDimension::kCube)) {
    return null_view;
  }

  TextureLayoutKey key = {};
  key.base_page = base_page;
  key.mip_page = mip_page;
  key.width_minus_1 = width_minus_1;
  key.height_minus_1 = height_minus_1;
  key.depth_or_array_size_minus_1 = depth_minus_1;
  key.pitch = fetch.pitch;
  key.mip_max_level = mip_max_level;
  key.format = uint32_t(format);
  key.dimension = uint32_t(fetch.dimension);
  key.endianness = uint32_t(fetch.endianness);
  key.tiled = fetch.tiled;
  key.packed_mips = fetch.packed_mips;
  uint64_t key_hash = XXH3_64bits(&key, sizeof(key));

  uint32_t width = width_minus_1 + 1;
  uint32_t height = height_minus_1 + 1;
  uint32_t layers = depth_minus_1 + 1;
  texture_util::TextureGuestLayout layout = texture_util::GetGuestTextureLayout(
      fetch.dimension, fetch.pitch, width, height, layers, fetch.tiled, format, fetch.packed_mips,
      base_page != 0, mip_max_level);

  const uint8_t* base_data =
      base_page ? im.memory->TranslatePhysical<const uint8_t*>(base_page << 12) : nullptr;
  const uint8_t* mip_data =
      mip_page ? im.memory->TranslatePhysical<const uint8_t*>(mip_page << 12) : nullptr;
  uint32_t base_size = base_page ? layout.base.level_data_extent_bytes : 0;
  uint32_t mip_size = mip_page ? layout.mips_total_extent_bytes : 0;
  if ((base_page << 12) + uint64_t(base_size) > 0x20000000u ||
      (mip_page << 12) + uint64_t(mip_size) > 0x20000000u) {
    return null_view;
  }

  // The content, hashed at most once per frame for each layout.
  Impl::Checked& checked = im.checked[key_hash];
  uint64_t texture_hash;
  if (checked.frame == im.frame) {
    texture_hash = checked.texture_hash;
  } else {
    uint64_t content = key_hash;
    if (base_size) {
      content ^= checked.base.Check(base_data, base_size, im.frame, key_hash, stats_.hash_bytes);
    }
    if (mip_size) {
      content ^= checked.mips.Check(mip_data, mip_size, im.frame, key_hash >> 8,
                                    stats_.hash_bytes) * 0x9E3779B97F4A7C15ull;
    }
    texture_hash = content;
    checked.frame = im.frame;
    checked.texture_hash = texture_hash;
  }
  const uint64_t valid_until =
      std::min(base_size ? checked.base.TrustedUntil(im.frame, key_hash) : UINT64_MAX,
               mip_size ? checked.mips.TrustedUntil(im.frame, key_hash >> 8) : UINT64_MAX);

  Texture* texture = nullptr;
  bool reuse_image = false;
  bool force_upload_now = false;
  auto existing = im.textures.find(key_hash);
  if (existing != im.textures.end() && existing->second->stream) {
    Texture& pending = *existing->second;
    StreamJob& job = *pending.stream;
    if (!job.done.load(std::memory_order_acquire)) {
      // Still streaming in: drawn without it for now.
      ++stats_.pending;
      return null_view;
    }
    if (!job.ok) {
      im.DestroyStreamJob(job);
      if (job.image) im.DestroyNow(job.image, job.memory, {});
      im.textures.erase(existing);
      return null_view;
    }
    // Ready: copy the staging data in with this submission.
    pending.image = job.image;
    pending.memory = job.memory;
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = pending.image;
    barrier.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
    dfn.vkCmdPipelineBarrier(upload.setup_commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);
    const UntileJob& u = job.untile;
    VkBufferImageCopy regions[xenos::kTextureMaxMips];
    for (uint32_t level = 0; level < u.level_count; ++level) {
      const LevelLayout& l = u.levels[level];
      VkBufferImageCopy& region = regions[level];
      region = {};
      region.bufferOffset = l.offset;
      region.bufferRowLength = l.blocks_x * u.block_width;
      region.bufferImageHeight = l.blocks_y * u.block_height;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, u.layers};
      region.imageExtent = {l.width, l.height, 1};
    }
    dfn.vkCmdCopyBufferToImage(upload.setup_commands, job.staging_buffer, pending.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, u.level_count, regions);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dfn.vkCmdPipelineBarrier(upload.setup_commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    // The staging memory goes once this frame has been copied.
    im.retired_staging.push_back({im.frame, pending.stream});
    pending.stream.reset();
    ++stats_.streamed;
  }
  if (existing != im.textures.end()) {
    Texture& old = *existing->second;
    if (old.content_hash == texture_hash) {
      texture = &old;
    } else if (old.last_used_submission < upload.submission) {
      texture = &old;
      reuse_image = true;
    } else {
      // Already sampled with the old content in this submission. The new
      // content goes into a new image, uploaded now rather than streamed:
      // streamed, the draws would have had no texture until it arrived, and
      // one rewritten every frame was missing most of the time.
      force_upload_now = true;
      ++stats_.replaced;
      std::vector<VkImageView> views;
      for (auto& [k, v] : old.views) views.push_back(v);
      im.Retire(old.image, old.memory, std::move(views), im.frame);
      im.textures.erase(existing);
      im.memo.clear();
      ++im.answers_epoch;
    }
  }
  if (!texture || reuse_image) {
    // Upload (into a new image, or the existing one).
    const FormatInfo* format_info = FormatInfo::Get(format);
    uint32_t block_width = format_info->block_width;
    uint32_t block_height = format_info->block_height;
    uint32_t guest_bytes_per_block = format_info->bytes_per_block();
    uint32_t bpb_log2 = rex::log2_floor(guest_bytes_per_block);
    bool host_blocks = host_format.compressed;
    // Bytes per host texel for uncompressed hosts (CTX1 decompresses 4x4).
    uint32_t host_texel_bytes = host_format.host_bytes_per_texel
                                    ? host_format.host_bytes_per_texel
                                    : guest_bytes_per_block;
    uint32_t level_count = mip_max_level + 1;
    if (!base_page) {
      // Only mips present: sample them as they are, the base is black.
      level_count = mip_max_level + 1;
    }

    // Host layout: every level and layer tightly packed, rows in whole blocks.
    level_count = std::min<uint32_t>(level_count, xenos::kTextureMaxMips);
    LevelLayout levels[xenos::kTextureMaxMips];
    VkDeviceSize total = 0;
    for (uint32_t level = 0; level < level_count; ++level) {
      LevelLayout& l = levels[level];
      l.width = std::max(width >> level, 1u);
      l.height = std::max(height >> level, 1u);
      l.blocks_x = (l.width + block_width - 1) / block_width;
      l.blocks_y = (l.height + block_height - 1) / block_height;
      if (host_blocks) {
        l.host_row_bytes = l.blocks_x * guest_bytes_per_block;
        l.host_rows = l.blocks_y;
      } else {
        // Whole guest blocks, so CTX1's 4x4 output never runs off the end.
        l.host_row_bytes = l.blocks_x * block_width * host_texel_bytes;
        l.host_rows = l.blocks_y * block_height;
      }
      l.layer_bytes = VkDeviceSize(l.host_row_bytes) * l.host_rows;
      l.offset = total;
      total += rex::align(l.layer_bytes * layers, VkDeviceSize(16));
    }

    // New large textures are created and untiled by the streaming worker, so
    // an area coming into view does not freeze a frame on hundreds of image
    // allocations (each a kernel round trip). They appear a frame or two
    // later.
    if (!reuse_image && !force_upload_now && im.sync_creates_left && total <= kSyncCreateMaxBytes) {
      --im.sync_creates_left;
      force_upload_now = true;
    }
    if (!reuse_image && !force_upload_now && total >= kStreamBytes) {
      auto job = std::make_shared<StreamJob>();
      UntileJob& u = job->untile;
      std::copy(levels, levels + level_count, u.levels);
      u.level_count = level_count;
      u.layers = layers;
      u.width = width;
      u.height = height;
      u.layout = layout;
      u.base_data = base_data;
      u.mip_data = mip_data;
      u.staging = nullptr;
      u.tiled = fetch.tiled;
      u.host_blocks = host_blocks;
      u.endianness = fetch.endianness;
      u.format = format;
      u.conversion = host_format.conversion;
      u.block_width = block_width;
      u.block_height = block_height;
      u.guest_bytes_per_block = guest_bytes_per_block;
      u.bpb_log2 = bpb_log2;
      u.host_texel_bytes = host_texel_bytes;
      job->staging_size = total;
      VkImageCreateInfo& info = job->info;
      info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = host_format.unorm;
      info.extent = {width, height, 1};
      info.mipLevels = level_count;
      info.arrayLayers = layers;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      if (fetch.dimension == xenos::DataDimension::kCube) {
        info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      }
      if (host_format.snorm != VK_FORMAT_UNDEFINED) {
        info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      }
      auto created = std::make_unique<Texture>();
      created->dimension = fetch.dimension;
      created->format_unorm = host_format.unorm;
      created->format_snorm = host_format.snorm;
      created->host_format_swizzle = host_format.swizzle;
      created->key_hash = key_hash;
      created->content_hash = texture_hash;
      created->level_count = level_count;
      created->layers = layers;
      created->last_used_frame = im.frame;
      created->stream = job;
      im.textures.emplace(key_hash, std::move(created));
      {
        std::lock_guard<std::mutex> lock(im.stream_mutex);
        im.stream_jobs.push_back(job);
      }
      im.stream_cv.notify_one();
      ++stats_.uploads;
      stats_.upload_bytes += total;
      im.CountUpload(key_hash, base_page << 12, uint32_t(format), width, height, total,
                     texture_hash);
      return null_view;
    }

    VkDeviceSize staging_offset = 0;
    uint8_t* staging = upload.allocate(size_t(total), 16, staging_offset);
    if (!staging) {
      return null_view;
    }

    // The untiling runs on the upload worker; the submission that copies the
    // staging data into the image waits for it (WaitForUploads).
    auto job = std::make_shared<UntileJob>();
    std::copy(levels, levels + level_count, job->levels);
    job->level_count = level_count;
    job->layers = layers;
    job->width = width;
    job->height = height;
    job->layout = layout;
    job->base_data = base_data;
    job->mip_data = mip_data;
    job->staging = staging;
    job->tiled = fetch.tiled;
    job->host_blocks = host_blocks;
    job->endianness = fetch.endianness;
    job->format = format;
    job->conversion = host_format.conversion;
    job->block_width = block_width;
    job->block_height = block_height;
    job->guest_bytes_per_block = guest_bytes_per_block;
    job->bpb_log2 = bpb_log2;
    job->host_texel_bytes = host_texel_bytes;
    im.EnqueueUpload([job]() { RunUntile(*job); });

    // The image.
    std::unique_ptr<Texture> created;
    if (!reuse_image) {
    created = std::make_unique<Texture>();
    created->dimension = fetch.dimension;
    created->format_unorm = host_format.unorm;
    created->format_snorm = host_format.snorm;
    created->host_format_swizzle = host_format.swizzle;
    created->key_hash = key_hash;
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = host_format.unorm;
    info.extent = {width, height, 1};
    info.mipLevels = level_count;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (fetch.dimension == xenos::DataDimension::kCube) {
      info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (host_format.snorm != VK_FORMAT_UNDEFINED) {
      info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }
    if (!rex::ui::vulkan::util::CreateDedicatedAllocationImage(
            im.device, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, created->image,
            created->memory)) {
      REXLOG_ERROR("[native] Could not create a {}x{}x{} texture of format {}", width, height,
                   layers, uint32_t(format));
      return null_view;
    }
    created->level_count = level_count;
    created->layers = layers;
    }
    Texture& target = reuse_image ? *texture : *created;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // A reused image may still be read by earlier submissions: the barrier's
    // first scope covers everything submitted before it on the queue.
    barrier.srcAccessMask = reuse_image ? VK_ACCESS_SHADER_READ_BIT : 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = target.image;
    barrier.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
    dfn.vkCmdPipelineBarrier(upload.setup_commands,
                             reuse_image ? (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                                         : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);
    VkBufferImageCopy regions[xenos::kTextureMaxMips];
    level_count = std::min<uint32_t>(level_count, xenos::kTextureMaxMips);
    for (uint32_t level = 0; level < level_count; ++level) {
      const LevelLayout& l = levels[level];
      VkBufferImageCopy& r = regions[level];
      r = {};
      r.bufferOffset = staging_offset + l.offset;
      r.bufferRowLength = l.blocks_x * block_width;
      r.bufferImageHeight = l.blocks_y * block_height;
      r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      r.imageSubresource.mipLevel = level;
      r.imageSubresource.baseArrayLayer = 0;
      r.imageSubresource.layerCount = layers;
      r.imageExtent = {l.width, l.height, 1};
    }
    dfn.vkCmdCopyBufferToImage(upload.setup_commands, upload.buffer, target.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, level_count, regions);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dfn.vkCmdPipelineBarrier(upload.setup_commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    ++stats_.uploads;
    stats_.upload_bytes += total;
    im.CountUpload(key_hash, base_page << 12, uint32_t(format), width, height, total,
                   texture_hash);
    target.content_hash = texture_hash;
    target.key_hash = key_hash;
    if (!reuse_image) {
      ++stats_.created;
      texture = created.get();
      im.textures.emplace(key_hash, std::move(created));
    }
  }

  texture->last_used_frame = im.frame;
  texture->last_used_submission = upload.submission;
  im.last_texture = texture;
  im.last_valid_until = valid_until;
  uint32_t swizzle = GuestToHostSwizzle(fetch.swizzle, texture->host_format_swizzle);
  bool use_signed = is_signed && texture->format_snorm != VK_FORMAT_UNDEFINED;
  uint32_t view_key = (swizzle << 1) | uint32_t(use_signed);
  auto view_it = texture->views.find(view_key);
  if (view_it != texture->views.end()) {
    return view_it->second;
  }
  VkImageViewCreateInfo view = {};
  view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view.image = texture->image;
  view.viewType = texture->dimension == xenos::DataDimension::kCube ? VK_IMAGE_VIEW_TYPE_CUBE
                                                                    : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view.format = use_signed ? texture->format_snorm : texture->format_unorm;
  view.components = {ComponentSwizzle(swizzle, 0), ComponentSwizzle(swizzle, 1),
                     ComponentSwizzle(swizzle, 2), ComponentSwizzle(swizzle, 3)};
  view.subresourceRange = rex::ui::vulkan::util::InitializeSubresourceRange();
  VkImageView result = VK_NULL_HANDLE;
  if (dfn.vkCreateImageView(d, &view, nullptr, &result) != VK_SUCCESS) {
    return null_view;
  }
  texture->views.emplace(view_key, result);
  return result;
}

VkSampler TextureCache::GetSampler(const xenos::xe_gpu_texture_fetch_t& fetch,
                                   const rex::graphics::SpirvShader::SamplerBinding& binding) {
  Impl& im = *impl_;
  // VulkanTextureCache::GetSamplerParameters and UseSampler in ReXGlue.
  xenos::ClampMode clamp_x, clamp_y, clamp_z;
  texture_util::GetClampModesForDimension(fetch, clamp_x, clamp_y, clamp_z);
  auto mag = binding.mag_filter == xenos::TextureFilter::kUseFetchConst ? fetch.mag_filter
                                                                        : binding.mag_filter;
  auto min = binding.min_filter == xenos::TextureFilter::kUseFetchConst ? fetch.min_filter
                                                                        : binding.min_filter;
  auto mip = binding.mip_filter == xenos::TextureFilter::kUseFetchConst ? fetch.mip_filter
                                                                        : binding.mip_filter;
  auto aniso = binding.aniso_filter == xenos::AnisoFilter::kUseFetchConst ? fetch.aniso_filter
                                                                          : binding.aniso_filter;
  uint32_t mip_min_level = 0, mip_max_level = 0;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr,
                                                 nullptr, &mip_min_level, &mip_max_level);
  bool mag_linear = mag == xenos::TextureFilter::kLinear;
  bool min_linear = min == xenos::TextureFilter::kLinear;
  bool mip_linear = mip == xenos::TextureFilter::kLinear;
  bool mip_base_map = mip == xenos::TextureFilter::kBaseMap;
  xenos::BorderColor border = xenos::ClampModeUsesBorder(clamp_x) ||
                                      xenos::ClampModeUsesBorder(clamp_y) ||
                                      xenos::ClampModeUsesBorder(clamp_z)
                                  ? fetch.border_color
                                  : xenos::BorderColor::k_ABGR_Black;
  if (aniso != xenos::AnisoFilter::kDisabled) {
    mag_linear = min_linear = mip_linear = true;
  }
  uint32_t key = uint32_t(clamp_x) | (uint32_t(clamp_y) << 3) | (uint32_t(clamp_z) << 6) |
                 (uint32_t(border) << 9) | (uint32_t(mag_linear) << 11) |
                 (uint32_t(min_linear) << 12) | (uint32_t(mip_linear) << 13) |
                 (uint32_t(aniso) << 14) | (mip_min_level << 17) | (uint32_t(mip_base_map) << 21);
  auto it = im.samplers.find(key);
  if (it != im.samplers.end()) {
    return it->second;
  }
  static const VkSamplerAddressMode kAddressModes[] = {
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
  };
  VkSamplerCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  info.minFilter = min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  info.mipmapMode = mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = kAddressModes[uint32_t(clamp_x) & 7];
  info.addressModeV = kAddressModes[uint32_t(clamp_y) & 7];
  info.addressModeW = kAddressModes[uint32_t(clamp_z) & 7];
  if (aniso != xenos::AnisoFilter::kDisabled && im.max_anisotropy > 1.0f) {
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = std::min(
        im.max_anisotropy,
        float(1u << (uint32_t(aniso) - uint32_t(xenos::AnisoFilter::kMax_1_1))));
  }
  info.minLod = float(mip_min_level);
  info.maxLod = mip_base_map ? info.minLod + 0.25f : VK_LOD_CLAMP_NONE;
  info.borderColor = border == xenos::BorderColor::k_ABGR_White
                         ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                         : (border == xenos::BorderColor::k_ABGR_Black
                                ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                                : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
  VkSampler sampler = VK_NULL_HANDLE;
  if (im.device->functions().vkCreateSampler(im.device->device(), &info, nullptr, &sampler) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  im.samplers.emplace(key, sampler);
  return sampler;
}

}  // namespace simpsons::native
