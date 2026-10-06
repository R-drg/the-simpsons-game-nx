// Cutscene video decoding.
//
// The story cutscenes are EA VP6 movies (movies/<language>/*.vp6), which the
// game decodes with its own copy of On2's VP6 decoder. A decode thread fills a
// queue of pictures ahead of time, and every frame the player shows the one
// that matches the clock of the movie's audio stream, dropping the late ones
// (sub_823733A0, sub_82375A18). VP6 cannot skip a picture it has not decoded,
// so when the decode thread produces fewer pictures a second than the movie
// has, the queue runs dry, the picture slows down and falls further and further
// behind the sound, which plays in real time. That is what cutscenes do on the
// Switch with the game's decoder, recompiled: at stock clocks they ran at 19
// FPS. With movie_decoder = "ffmpeg", FFmpeg's VP6 decoder decodes the same
// packets into the same buffers.
//
// The game's decoder (its state, On2's PB_INSTANCE, is "pbi" below):
//  - sub_8237A1F0(pbi, data, size) decodes one picture. The movie codec's
//    decode method (sub_82379DF0) calls it with the payload of an MV0K or MV0F
//    chunk, the 8-byte chunk header skipped. It returns 0, or -1 for no new
//    picture.
//  - It writes the picture straight into the player's buffers: Y, U and V at
//    [pbi+17236], [pbi+17240] and [pbi+17244], with the luma and chroma
//    strides at [pbi+17248] and [pbi+17252], bitstream row r at plane +
//    r * stride. The caller then reads the coded size (16 pixels per
//    macroblock) back from [pbi+332] and [pbi+336].
//  - sub_8237A5C8(&pbi) frees it.
// FFmpeg's "vp6" decoder, the one it uses for EA movies, returns the picture
// upside down with respect to the bitstream rows, so the rows are copied in
// reverse. Both were checked on a host build of the SDK's FFmpeg with an EA VP6
// sample: packets cut like the game's decode bit for bit like FFmpeg's own
// demuxer, and the reversed rows are FFmpeg's unflipped "vp6f" output exactly.
//
// movie_decoder = "check" keeps the game's decoder, also decodes every packet
// with FFmpeg and logs how much the two pictures differ, per movie.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ppc.h>

extern "C" {
#include "libavcodec/avcodec.h"
}

REXCVAR_DEFINE_STRING(movie_decoder, "ffmpeg", "Game",
                      "Decoder of the cutscene videos: ffmpeg = FFmpeg's VP6 decoder; game = the "
                      "game's own, recompiled (too slow on the Switch: the picture falls behind the "
                      "sound); check = the game's, compared with FFmpeg's in the log")
    .allowed({"ffmpeg", "game", "check"});

// Decodes one VP6 picture: r3 = pbi, r4 = the packet, r5 = its size (from the
// movie codec's decode method).
REX_EXTERN(__imp__sub_8237A1F0);
REX_EXTERN(sub_8237A1F0);

// Frees the decoder: r3 = the address of the pbi pointer.
REX_EXTERN(__imp__sub_8237A5C8);
REX_EXTERN(sub_8237A5C8);

namespace {

// Return address of the call in sub_82379DF0, the one caller whose r5 is the
// packet's size. sub_8237A378 jumps to the decoder too, with arguments of its own.
constexpr uint32_t kMovieDecodeReturn = 0x82379EA4;

constexpr uint32_t kPbiCodedWidth = 332;
constexpr uint32_t kPbiCodedHeight = 336;
constexpr uint32_t kPbiOutputY = 17236;
constexpr uint32_t kPbiOutputU = 17240;
constexpr uint32_t kPbiOutputV = 17244;
constexpr uint32_t kPbiOutputYStride = 17248;
constexpr uint32_t kPbiOutputUVStride = 17252;

// Far above any movie's largest chunk; a bigger "size" is not a packet.
constexpr uint32_t kMaxPacketBytes = 8u << 20;

enum class Mode { kFFmpeg, kGame, kCheck };

uint32_t LoadBE32(const uint8_t* base, uint32_t address) {
  uint32_t value;
  std::memcpy(&value, base + address, sizeof(value));
  return rex::byte_swap(value);
}

void StoreBE32(uint8_t* base, uint32_t address, uint32_t value) {
  value = rex::byte_swap(value);
  std::memcpy(base + address, &value, sizeof(value));
}

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// One movie: a game decoder and the FFmpeg decoder that replaces it.
struct Movie {
  Mode mode = Mode::kFFmpeg;
  AVCodecContext* codec = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  std::vector<uint8_t> data;  // the packet, with the zeroed padding FFmpeg reads past its end
  int width = 0;
  int height = 0;

  uint64_t decoded = 0;  // pictures from FFmpeg
  uint64_t rejected = 0;  // packets FFmpeg could not decode
  int64_t ffmpeg_ns = 0;
  int64_t ffmpeg_max_ns = 0;

  // movie_decoder = "check"
  uint64_t game_decoded = 0;
  int64_t game_ns = 0;
  uint64_t compared = 0;
  uint64_t differing = 0;
  int max_difference = 0;
  double mean_difference_sum = 0.0;
  double straight_mean_difference = -1.0;  // first picture, rows not reversed

  ~Movie() {
    avcodec_free_context(&codec);
    av_frame_free(&frame);
    av_packet_free(&packet);
  }
};

std::mutex g_movies_mutex;
std::unordered_map<uint32_t, std::unique_ptr<Movie>> g_movies;  // by pbi

bool OpenFFmpeg(Movie& movie) {
  const AVCodec* vp6 = avcodec_find_decoder(AV_CODEC_ID_VP6);
  if (!vp6) {
    return false;
  }
  movie.codec = avcodec_alloc_context3(vp6);
  movie.frame = av_frame_alloc();
  movie.packet = av_packet_alloc();
  if (!movie.codec || !movie.frame || !movie.packet) {
    return false;
  }
  movie.codec->thread_count = 1;
  return avcodec_open2(movie.codec, vp6, nullptr) >= 0;
}

Movie& FindMovie(uint32_t pbi) {
  auto& movie = g_movies[pbi];
  if (movie) {
    return *movie;
  }
  movie = std::make_unique<Movie>();
  const std::string mode = REXCVAR_GET(movie_decoder);
  movie->mode = mode == "game" ? Mode::kGame : mode == "check" ? Mode::kCheck : Mode::kFFmpeg;
  if (movie->mode != Mode::kGame && !OpenFFmpeg(*movie)) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      REXLOG_WARN("Movie decoder: FFmpeg's VP6 decoder is not available; the game's decodes");
    }
    movie->mode = Mode::kGame;
  }
  return *movie;
}

// Decodes a packet into movie.frame.
bool DecodeWithFFmpeg(Movie& movie, const uint8_t* data, uint32_t size) {
  const int64_t start = NowNs();
  movie.data.assign(data, data + size);
  movie.data.resize(size_t(size) + AV_INPUT_BUFFER_PADDING_SIZE, 0);
  movie.packet->data = movie.data.data();
  movie.packet->size = int(size);
  const bool decoded = avcodec_send_packet(movie.codec, movie.packet) >= 0 &&
                       avcodec_receive_frame(movie.codec, movie.frame) >= 0 &&
                       movie.frame->format == AV_PIX_FMT_YUV420P;
  const int64_t elapsed = NowNs() - start;
  movie.ffmpeg_ns += elapsed;
  movie.ffmpeg_max_ns = std::max(movie.ffmpeg_max_ns, elapsed);
  if (!decoded) {
    ++movie.rejected;
    return false;
  }
  ++movie.decoded;
  if (movie.frame->width != movie.width || movie.frame->height != movie.height) {
    movie.width = movie.frame->width;
    movie.height = movie.frame->height;
    REXLOG_INFO("Movie decoder: {}x{} VP6, decoded by {}", movie.width, movie.height,
                movie.mode == Mode::kCheck ? "the game, checked against FFmpeg" : "FFmpeg");
  }
  return true;
}

// Guest row r of a plane is at dest + r * stride, in 32-bit guest addresses
// like the game's own stores; FFmpeg's rows are bottom-up.
void CopyPlane(uint8_t* base, uint32_t dest, int32_t stride, const uint8_t* source,
               int source_stride, int width, int height) {
  if (!dest) {
    return;
  }
  for (int row = 0; row < height; ++row) {
    const uint32_t address = dest + uint32_t(int64_t(row) * stride);
    std::memcpy(base + address, source + ptrdiff_t(height - 1 - row) * source_stride,
                size_t(width));
  }
}

void WritePicture(uint8_t* base, uint32_t pbi, const AVFrame& frame) {
  const int32_t y_stride = int32_t(LoadBE32(base, pbi + kPbiOutputYStride));
  const int32_t uv_stride = int32_t(LoadBE32(base, pbi + kPbiOutputUVStride));
  const int width = frame.width;
  const int height = frame.height;
  CopyPlane(base, LoadBE32(base, pbi + kPbiOutputY), y_stride, frame.data[0], frame.linesize[0],
            width, height);
  CopyPlane(base, LoadBE32(base, pbi + kPbiOutputU), uv_stride, frame.data[1], frame.linesize[1],
            width / 2, height / 2);
  CopyPlane(base, LoadBE32(base, pbi + kPbiOutputV), uv_stride, frame.data[2], frame.linesize[2],
            width / 2, height / 2);
  StoreBE32(base, pbi + kPbiCodedWidth, uint32_t(width));
  StoreBE32(base, pbi + kPbiCodedHeight, uint32_t(height));
}

// Mean absolute difference between the luma the game wrote and FFmpeg's.
double LumaDifference(const uint8_t* base, uint32_t pbi, const AVFrame& frame, bool reversed,
                      int* max_difference) {
  const uint32_t dest = LoadBE32(base, pbi + kPbiOutputY);
  const int32_t stride = int32_t(LoadBE32(base, pbi + kPbiOutputYStride));
  uint64_t sum = 0;
  int max = 0;
  for (int row = 0; row < frame.height; ++row) {
    const uint8_t* game = base + (dest + uint32_t(int64_t(row) * stride));
    const uint8_t* ffmpeg =
        frame.data[0] + ptrdiff_t(reversed ? frame.height - 1 - row : row) * frame.linesize[0];
    for (int x = 0; x < frame.width; ++x) {
      const int difference = std::abs(int(game[x]) - int(ffmpeg[x]));
      sum += uint64_t(difference);
      max = std::max(max, difference);
    }
  }
  if (max_difference) {
    *max_difference = max;
  }
  return double(sum) / double(std::max(1, frame.width * frame.height));
}

void Compare(Movie& movie, const uint8_t* base, uint32_t pbi) {
  if (!LoadBE32(base, pbi + kPbiOutputY)) {
    return;
  }
  int max = 0;
  const double mean = LumaDifference(base, pbi, *movie.frame, true, &max);
  if (movie.compared == 0) {
    movie.straight_mean_difference = LumaDifference(base, pbi, *movie.frame, false, nullptr);
  }
  ++movie.compared;
  movie.differing += max != 0;
  movie.max_difference = std::max(movie.max_difference, max);
  movie.mean_difference_sum += mean;
}

void Report(const Movie& movie) {
  if (movie.mode == Mode::kGame || (!movie.decoded && !movie.rejected)) {
    return;
  }
  const double ffmpeg_ms = movie.decoded ? double(movie.ffmpeg_ns) / 1e6 / double(movie.decoded) : 0.0;
  REXLOG_INFO("Movie decoder: {}x{}, {} pictures from FFmpeg, {:.2f} ms each on average, "
              "{:.2f} ms at most, {} packets rejected",
              movie.width, movie.height, movie.decoded, ffmpeg_ms,
              double(movie.ffmpeg_max_ns) / 1e6, movie.rejected);
  if (movie.mode == Mode::kCheck && movie.game_decoded) {
    REXLOG_INFO("Movie decoder check: the game's decoder took {:.2f} ms a picture; {} of {} "
                "pictures differ from FFmpeg's, mean luma difference {:.4f}, at most {} (rows "
                "not reversed, first picture: {:.2f})",
                double(movie.game_ns) / 1e6 / double(movie.game_decoded), movie.differing,
                movie.compared,
                movie.compared ? movie.mean_difference_sum / double(movie.compared) : 0.0,
                movie.max_difference, movie.straight_mean_difference);
  }
}

}  // namespace

REX_FUNC(sub_8237A1F0) {
  const uint32_t pbi = ctx.r3.u32;
  const uint32_t data = ctx.r4.u32;
  const uint32_t size = ctx.r5.u32;
  if (ctx.lr != kMovieDecodeReturn || !size || size > kMaxPacketBytes ||
      uint64_t(data) + size > 0x100000000ull) {
    __imp__sub_8237A1F0(ctx, base);
    return;
  }
  // The movie's decode thread is the only caller for its pbi; the lock is for
  // the map, which other movies and sub_8237A5C8 change.
  std::lock_guard<std::mutex> lock(g_movies_mutex);
  Movie& movie = FindMovie(pbi);
  switch (movie.mode) {
    case Mode::kGame:
      __imp__sub_8237A1F0(ctx, base);
      return;
    case Mode::kCheck: {
      const int64_t start = NowNs();
      __imp__sub_8237A1F0(ctx, base);
      movie.game_ns += NowNs() - start;
      ++movie.game_decoded;
      const bool game_decoded = ctx.r3.u32 == 0;
      if (DecodeWithFFmpeg(movie, base + data, size) && game_decoded) {
        Compare(movie, base, pbi);
      }
      return;
    }
    case Mode::kFFmpeg:
      if (!DecodeWithFFmpeg(movie, base + data, size)) {
        // Like the game's decoder with a header it cannot read: no new
        // picture. The next key frame starts over.
        ctx.r3.s64 = -1;
        return;
      }
      WritePicture(base, pbi, *movie.frame);
      ctx.r3.u64 = 0;
      return;
  }
}

REX_FUNC(sub_8237A5C8) {
  const uint32_t pbi = LoadBE32(base, ctx.r3.u32);
  std::unique_ptr<Movie> movie;
  if (pbi) {
    std::lock_guard<std::mutex> lock(g_movies_mutex);
    auto it = g_movies.find(pbi);
    if (it != g_movies.end()) {
      movie = std::move(it->second);
      g_movies.erase(it);
    }
  }
  if (movie) {
    Report(*movie);
  }
  __imp__sub_8237A5C8(ctx, base);
}
