#include <array>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "stb_image_write.h"

void panic(const std::string &error) {
  fprintf(stderr, "%s\n", error.c_str());
  exit(1);
}

template <int WIDTH, int HEIGHT> struct Image {
  std::array<uint8_t, WIDTH * HEIGHT> data;

  inline void set(int x, int y, uint8_t value) { data[y * WIDTH + x] = value; }
};

template <int WIDTH, int HEIGHT>
void save_image(char const *filename, const Image<WIDTH, HEIGHT> &img) {
  int ret = stbi_write_png(filename, WIDTH, HEIGHT, 1, &img.data[0], WIDTH);
  if (ret == 0) {
    panic("stbi_write_png failed (Error saving image file)");
  }
}

template <int WIDTH, int HEIGHT>
void fill_rect(Image<WIDTH, HEIGHT> &img, int x1, int y1, int x2, int y2,
               uint8_t color) {
  for (int y = y1; y < y2; ++y) {
    for (int x = x1; x < x2; ++x) {
      img.set(x, y, color);
    }
  }
}

const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;

// RAW_FRAME_HEIGHT seems to be different on different hardware:

// raspberrypi-1, raspberrypi-2
const int RAW_FRAME_HEIGHT = 1080;

// raspberrypi-4
// const int RAW_FRAME_HEIGHT = 720;

const int OUTPUT_WIDTH = 320;
const int OUTPUT_HEIGHT = 180;

void read_full_frame(FILE *fp, Image<FRAME_WIDTH, FRAME_HEIGHT> &out) {
  size_t ret = fread(&out.data, FRAME_WIDTH * FRAME_HEIGHT, 1, fp);
  if (ret != 1) {
    panic("fread error");
  }
}

void skip_frame(FILE *fp) {
  const int CHUNK_WIDTH = FRAME_WIDTH;
  const int CHUNK_HEIGHT = 4;

  const int NUM_CHUNKS = FRAME_HEIGHT / CHUNK_HEIGHT;
  static_assert(NUM_CHUNKS == OUTPUT_HEIGHT);

  uint8_t chunk[CHUNK_WIDTH * CHUNK_HEIGHT];

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    size_t ret = fread(chunk, CHUNK_WIDTH * CHUNK_HEIGHT, 1, fp);
    if (ret != 1) {
      panic("fread error");
    }
  }

  // Read in the rest of the padding buffer
  const int PADDING_NUM_CHUNKS =
      (RAW_FRAME_HEIGHT - FRAME_HEIGHT) / CHUNK_HEIGHT;
  static_assert(PADDING_NUM_CHUNKS * CHUNK_HEIGHT ==
                RAW_FRAME_HEIGHT - FRAME_HEIGHT);
  for (int i = 0; i < PADDING_NUM_CHUNKS; ++i) {
    size_t ret = fread(chunk, CHUNK_WIDTH * CHUNK_HEIGHT, 1, fp);
    if (ret != 1) {
      panic("fread error");
    }
  }
}

void read_scaled_frame(FILE *fp, Image<OUTPUT_WIDTH, OUTPUT_HEIGHT> &out) {
  // Read 4 lines at a time (don't use fread, use direct reading to bypass
  // buffering, actually, maybe buffering is good!)

  const int CHUNK_WIDTH = FRAME_WIDTH;
  const int CHUNK_HEIGHT = 4;

  const int NUM_CHUNKS = FRAME_HEIGHT / CHUNK_HEIGHT;
  static_assert(NUM_CHUNKS == OUTPUT_HEIGHT);

  uint8_t chunk[CHUNK_WIDTH * CHUNK_HEIGHT];

  for (int i = 0; i < NUM_CHUNKS; ++i) {
    size_t ret = fread(chunk, CHUNK_WIDTH * CHUNK_HEIGHT, 1, fp);
    if (ret != 1) {
      panic("fread error");
    }
    for (int x = 0; x < OUTPUT_WIDTH; ++x) {
      int sum = 0;
      for (int v = 0; v < 4; ++v) {
        for (int u = 0; u < 4; ++u) {
          sum += static_cast<int>(chunk[v * CHUNK_WIDTH + u + x * 4]);
        }
      }
      out.set(x, i, sum / 16);
    }
  }

  // Erase the timestamp
  fill_rect(out, 128, 5, 192, 13, 0);

  // Read in the rest of the padding buffer
  const int PADDING_NUM_CHUNKS =
      (RAW_FRAME_HEIGHT - FRAME_HEIGHT) / CHUNK_HEIGHT;
  static_assert(PADDING_NUM_CHUNKS * CHUNK_HEIGHT ==
                RAW_FRAME_HEIGHT - FRAME_HEIGHT);
  for (int i = 0; i < PADDING_NUM_CHUNKS; ++i) {
    size_t ret = fread(chunk, CHUNK_WIDTH * CHUNK_HEIGHT, 1, fp);
    if (ret != 1) {
      panic("fread error");
    }
  }
}

// Returns unix epoch in milliseconds
int64_t get_current_time() {
  struct timespec spec;

  if (clock_gettime(CLOCK_REALTIME, &spec) != 0) {
    panic("Error getting time");
  }

  return (static_cast<int64_t>(spec.tv_sec) * static_cast<int64_t>(1000)) +
         (static_cast<int64_t>(spec.tv_nsec) / static_cast<int64_t>(1000000));
}

void frame_loop(FILE *fp) {
  Image<320, 180> img;

  int frame_num = 0;
  while (true) {
    printf("frame_num %d\n", frame_num);
    int64_t timestamp = get_current_time();
    // if (frame_num % 450 == 0 || frame_num % 450 == 3 || frame_num % 450 == 6) {
    if (frame_num % 4 == 0) {
      read_scaled_frame(fp, img);

      char filename[256];
      snprintf(filename, sizeof(filename), "out/frame-%09d-%" PRId64 ".png",
               frame_num, timestamp);

      const char *tmp_file = "out/tmp.png";
      save_image(tmp_file, img);
      rename(tmp_file, filename);
    } else {
      skip_frame(fp);
    }

    frame_num++;
  }
}

void test_save_two_frames() {
  FILE *fp = fopen("foo", "rb");
  // fseek(fp, 30 * FRAME_WIDTH * RAW_FRAME_HEIGHT, SEEK_CUR);

  Image<320, 180> img;
  // read_full_frame(fp, img);
  read_scaled_frame(fp, img);
  save_image("image1.png", img);
  read_scaled_frame(fp, img);
  save_image("image2.png", img);
  fclose(fp);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s INPUT_FILE\n", argv[0]);
    return 1;
  }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) {
    perror((std::string("Error opening file ") + argv[1]).c_str());
    exit(1);
  }

  frame_loop(fp);

  return 0;
}
