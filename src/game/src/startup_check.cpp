// Startup image check.
//
// The first Switch run crashed in the game's first function after its static
// constructors, walking a list whose head (0x82CD272C) held a pointer that is
// not one. These lines say whether the loaded XEX image already differs from
// the PC build's before any guest code runs (a loader problem), or only by the
// time that function runs (something overwrote it). Two log lines per boot.

#include <cstdint>
#include <cstring>

#include <rex/logging.h>
#include <rex/ppc.h>

namespace {

constexpr uint32_t kImageStart = 0x82000000;
constexpr uint32_t kImageEnd = 0x82EC0000;
constexpr uint32_t kListHead = 0x82CD272C;

uint32_t LoadBE32(const uint8_t* base, uint32_t address) {
  uint32_t value;
  std::memcpy(&value, base + address, sizeof(value));
  return rex::byte_swap(value);
}

uint64_t Fnv1a(const uint8_t* data, size_t size) {
  uint64_t hash = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * 0x100000001B3ull;
  }
  return hash;
}

void LogImage(const char* when, const uint8_t* base) {
  // The data sections only: the code is never read from guest memory.
  constexpr uint32_t kDataStart = 0x82CC4000;
  REXLOG_INFO("startup check ({}): image hash {:016X}, data hash {:016X}, list head {:08X} {:08X}",
              when, Fnv1a(base + kImageStart, kImageEnd - kImageStart),
              Fnv1a(base + kDataStart, kImageEnd - kDataStart), LoadBE32(base, kListHead),
              LoadBE32(base, kListHead + 4));
}

}  // namespace

namespace simpsons {

void LogLoadedImage(const uint8_t* base) {
  LogImage("loaded", base);
}

}  // namespace simpsons

// The function that crashed: walks the list at kListHead under its critical
// section, calling each entry.
REX_EXTERN(__imp__sub_82433FD0);
REX_EXTERN(sub_82433FD0);

REX_FUNC(sub_82433FD0) {
  static bool logged = false;
  if (!logged) {
    logged = true;
    LogImage("first list walk", base);
  }
  __imp__sub_82433FD0(ctx, base);
}
