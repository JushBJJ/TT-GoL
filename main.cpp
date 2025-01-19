#include "host_api.hpp"
#include "impl/device/device.hpp"
#include "common/bfloat16.hpp"
#include "common/logger.hpp"
#include "common/constants.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <cstring>
#include <SDL2/SDL.h>
#include <map>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

float int32_to_float(uint32_t val) {
    float result;
    std::memcpy(&result, &val, sizeof(float));
    return result;
}

std::map<std::string, std::vector<std::pair<int, int>>> lexicons = {
    {"glider", {{1, 0}, {2, 1}, {0, 2}, {1, 2}, {2, 2}}},
    {"lwss", {{0, 0}, {3, 0}, {4, 1}, {0, 2}, {4, 2}, {1, 3}, {2, 3}, {3, 3}, {4, 3}}},
};

void insert_lexicon(std::vector<int>& top_vec, std::vector<int>& bottom_vec, const std::string& lexicon_name, int padded_width, int padded_height) {
    if (lexicons.find(lexicon_name) == lexicons.end()) {
        std::cerr << "Lexicon not found: " << lexicon_name << std::endl;
        return;
    }

    auto& lexicon = lexicons[lexicon_name];
    int grid_center_y = padded_height / 2;
    int grid_center_x = padded_width / 2;

    for (const auto& coord : lexicon) {
        int y = grid_center_y + coord.second - lexicon.size() / 2;
        int x = grid_center_x + coord.first - lexicon.size() / 2;

        if (y >= 0 && y < padded_height && x >= 0 && x < padded_width) {
            if (y < padded_height / 2) {
                top_vec[y * padded_width + x] = 1;
            } else {
                bottom_vec[(y - padded_height / 2) * padded_width + x] = 1;
            }
        }
    }
}

void insert_random_grid(std::vector<int>& top_vec, std::vector<int>& bottom_vec, int padded_width, int padded_height, float density) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (int y = 0; y < padded_height; y++) {
        for (int x = 0; x < padded_width; x++) {
            if (dist(gen) < density) {
                if (y < padded_height / 2) {
                    top_vec[y * padded_width + x] = 1;
                } else {
                    bottom_vec[(y - padded_height / 2) * padded_width + x] = 1;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return 1;
    }

    int datatype_size = sizeof(uint16_t);
    int height = 64;
    int width = 64;
    int num_tiles = 2;
    int padded_height = ((height + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    int padded_width = ((width + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;

    uint32_t single_tile_size = datatype_size * padded_width * padded_height;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    CoreCoord core = {0, 0};
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> top_vec(padded_width * padded_height, 0);
    std::vector<int> bottom_vec(padded_width * padded_height, 0);
    std::vector<int> output_vec(padded_width * padded_height, 0);

    std::string spawn_type = "random";
    if (spawn_type == "random") {
        float density = 0.5;
        insert_random_grid(top_vec, bottom_vec, padded_width, padded_height, density);
    }
    else {
        std::string lexicon_to_spawn = spawn_type;
        insert_lexicon(top_vec, bottom_vec, lexicon_to_spawn, padded_width, padded_height);
    }

    Device *device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    InterleavedBufferConfig dram_config {
        .device = device,
        .size = dram_buffer_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };

    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> src2_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst0_dram_buffer = CreateBuffer(dram_config);

    uint32_t src0_bank_id = 0;
    uint32_t src1_bank_id = 0;
    uint32_t src2_bank_id = 0;
    uint32_t dst0_bank_id = 0;

    constexpr uint32_t src0_cb_index = CB::c_in0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        single_tile_size * 2,
        {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CB::c_in1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(
        single_tile_size * 2,
        {{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t src2_cb_index = CB::c_in2;
    CircularBufferConfig cb_src2_config = CircularBufferConfig(
        single_tile_size * 2,
        {{src2_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src2_cb_index, single_tile_size);
    CBHandle cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    constexpr uint32_t dst0_cb_index = CB::c_out0;
    CircularBufferConfig cb_dst0_config = CircularBufferConfig(
        single_tile_size * 2,
        {{dst0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(dst0_cb_index, single_tile_size);
    CBHandle cb_dst0 = tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);
    
    KernelHandle reader = CreateKernel(
        program,
        "./kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    KernelHandle writer = CreateKernel(
        program,
        "./kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    SetRuntimeArgs(program, reader, core,
        {
            src0_dram_buffer->address(),
            src0_bank_id,
            num_tiles,

            src1_dram_buffer->address(),
            src1_bank_id,

            src2_dram_buffer->address(),
            src2_bank_id,
        }
    );

    SetRuntimeArgs(program, writer, core,
        {
            dst0_dram_buffer->address(),
            dst0_bank_id,
            num_tiles
        }
    );

    int num_frames = 5000;
    std::vector<uint32_t> output(padded_width * padded_height, 0);

    int window_width = 512;
    int window_height = 512;
    int scale_factor = window_width / width;

    SDL_Window *window = SDL_CreateWindow("Output Visualization", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, window_width, window_height, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);

    int i = 0;
    bool quit = false;
    const int frame_delay_ms = 50;
    SDL_Event event;

    while (!quit) {
        if (i < num_frames) {
            std::cout << "Frame " << i + 1 << " of " << num_frames << "\n";
            EnqueueWriteBuffer(cq, src0_dram_buffer, top_vec, true);
            EnqueueWriteBuffer(cq, src1_dram_buffer, bottom_vec, true);
            EnqueueWriteBuffer(cq, src2_dram_buffer, output_vec, true);
            EnqueueProgram(cq, program, true);
            EnqueueReadBuffer(cq, dst0_dram_buffer, output, true);
            i++;

            for (int j = 0; j < padded_height; j++) {
                for (int k = 0; k < padded_width; k++) {
                    if (j < padded_height / 2) {
                        top_vec[j * padded_width + k] = output[j * padded_width + k];
                    } else {
                        bottom_vec[(j - (padded_height / 2)) * padded_width + k] = output[j * padded_width + k];
                    }
                }
            }

            uint32_t *pixels;
            int pitch;
            SDL_LockTexture(texture, NULL, (void **)&pixels, &pitch);

            for (int i = 0; i < width * height; i++) {
                pixels[i] = output[i] == 0 ? 0xFF000000 : 0xFFFFFFFF;
            }

            SDL_UnlockTexture(texture);
            SDL_Rect destRect = {0, 0, window_width, window_height};
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, &destRect);
            SDL_RenderPresent(renderer);
            SDL_Delay(frame_delay_ms);
        }

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_q) {
                    quit = true;
                }
            }
        }
    }

    std::cout << std::endl;

    Finish(cq);
    CloseDevice(device);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}