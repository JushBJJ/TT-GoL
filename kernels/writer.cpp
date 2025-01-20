#include "dataflow_api.h"

// throw in quake's square root because we cant do #include <cmath> usually
// and because why not? :P
// this one is a bit modified so it can comply to the RISC-V compiler
float Q_rsqrt(float number)
{
    float x2 = number * 0.5F;
    float y = number;
    const float threehalfs = 1.5F;
    
    union {
        float f;
        int32_t i;
    } conv;
    
    conv.f = number;                        // evil floating point bit level hacking
    conv.i = 0x5f3759df - (conv.i >> 1);   // what the fuck?
    y = conv.f;
    
    y = y * (threehalfs - (x2 * y * y));   // 1st iteration
    // y = y * (threehalfs - (x2 * y * y)); // 2nd iteration, can be removed

    return y;
}

float Q_sqrt(float n) {
    return n * Q_rsqrt(n);
}

namespace GoL {
    inline int to_idx(int x, int y, int tile_width, int tile_height) {
        x = ((x % tile_width) + tile_width) % tile_width;
        y = ((y % tile_height) + tile_height) % tile_height;
        return y * tile_width + x;
    }

    inline int count_neighbours(const uint32_t* top_tile, 
                              const uint32_t* bottom_tile,
                              int idx,
                              int tile_width,
                              int tile_height,
                              int single_tile_size,
                              bool second_tile) {
        int count = 0;
        int x = idx % tile_width;
        int y = idx / tile_width;

        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                int n_idx = to_idx(x + i, y + j, tile_width, tile_height);
                if (n_idx < single_tile_size) {
                    count += top_tile[n_idx];
                } else {
                    count += bottom_tile[n_idx - single_tile_size];
                }
            }
        }
        return count;
    }

    void step(const uint32_t* top_tile,
            const uint32_t* bottom_tile,
            uint32_t* output_tile,
            int tile_width,
            int tile_height,
            int single_tile_size,
            bool second_tile) {
        const uint32_t* current_tile = second_tile ? bottom_tile : top_tile;

        for (int y = 0; y < tile_height / 2; y++) {
            for (int x = 0; x < tile_width; x++) {
                int real_y = second_tile ? y + tile_height / 2: y;
                int real_idx = real_y * tile_width + x;
                int idx = y * tile_width + x;
                int count = count_neighbours(top_tile, bottom_tile, real_idx, tile_width, tile_height, single_tile_size, second_tile);
                uint32_t cell = current_tile[idx];

                if (cell == 1) {
                    if (count < 2 || count > 3) {
                        cell = 0;
                    }
                } else {
                    if (count == 3) {
                        cell = 1;
                    }
                }

                output_tile[idx] = cell;
            }
        }
    }
}

void kernel_main() {
    constexpr bool is_dram = true;
    constexpr int single_tile = 2;

    constexpr auto cb_id_in0 = tt::CB::c_in0;
    constexpr auto cb_id_in1 = tt::CB::c_in1;
    constexpr auto cb_id_in2 = tt::CB::c_in2;

    auto dst0_dram_addr = get_arg_val<uint32_t>(0);
    auto dst0_bank_id = get_arg_val<uint32_t>(1);
    auto num_tiles = get_arg_val<uint32_t>(2);

    uint64_t dst0_dram_noc_addr = get_noc_addr_from_bank_id<is_dram>(dst0_bank_id, dst0_dram_addr);

    const uint32_t dst0_tile_size = get_tile_size(cb_id_in2) * 4;
    const DataFormat dst0_data_format = get_dataformat(cb_id_in2);

    const InterleavedAddrGenFast<is_dram> s0 = {
        .bank_base_address = dst0_dram_addr,
        .page_size = dst0_tile_size,
        .data_format = dst0_data_format
    };

    cb_wait_front(cb_id_in0, single_tile);
    cb_wait_front(cb_id_in1, single_tile);

    uint32_t top = get_write_ptr(cb_id_in0);
    uint32_t bottom = get_write_ptr(cb_id_in1);

    uint32_t *top_tile = (uint32_t *)top;
    uint32_t *bottom_tile = (uint32_t *)bottom;

    cb_pop_front(cb_id_in0, single_tile);
    cb_pop_front(cb_id_in1, single_tile);

    bool once = false;
    int single_tile_size = dst0_tile_size / 4;
    int tile_width = (int)(Q_sqrt(single_tile_size * 2) + 0.5f);
    int tile_height = tile_width;

    // equilvalent of a single step
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_in2, single_tile);

        uint32_t output = get_write_ptr(cb_id_in2);
        uint32_t *output_tile = (uint32_t *)output;

        GoL::step(top_tile, bottom_tile, output_tile, tile_width, tile_height, single_tile_size, once);

        once = true;

        noc_async_write_tile(i, s0, output);
        noc_async_write_barrier();

        cb_pop_front(cb_id_in2, single_tile);
    }
}