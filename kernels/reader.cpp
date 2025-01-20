#include "dataflow_api.h"

void kernel_main() {
    constexpr bool is_dram = true;
    constexpr int single_tile = 2;

    constexpr auto cb_id_in0 = tt::CB::c_in0;
    constexpr auto cb_id_in1 = tt::CB::c_in1;
    constexpr auto cb_id_in2 = tt::CB::c_in2;

    auto src0_dram_addr = get_arg_val<uint32_t>(0);
    auto src0_bank_id = get_arg_val<uint32_t>(1);
    auto num_tiles = get_arg_val<uint32_t>(2);
    auto src1_dram_addr = get_arg_val<uint32_t>(3);
    auto src1_bank_id = get_arg_val<uint32_t>(4);
    auto src2_dram_addr = get_arg_val<uint32_t>(5);
    auto src2_bank_id = get_arg_val<uint32_t>(6);

    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<is_dram>(src0_bank_id, src0_dram_addr);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<is_dram>(src1_bank_id, src1_dram_addr);
    uint64_t src2_dram_noc_addr = get_noc_addr_from_bank_id<is_dram>(src2_bank_id, src2_dram_addr);

    const uint32_t src0_tile_size = get_tile_size(cb_id_in0) * 4;
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const uint32_t src1_tile_size = get_tile_size(cb_id_in1) * 4;
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);
    const uint32_t src2_tile_size = get_tile_size(cb_id_in2) * 4;
    const DataFormat src2_data_format = get_dataformat(cb_id_in2);

    const InterleavedAddrGenFast<is_dram> s0 = {
        .bank_base_address = src0_dram_addr,
        .page_size =  src0_tile_size,
        .data_format =  src0_data_format
    };

    const InterleavedAddrGenFast<is_dram> s1 = {
        .bank_base_address = src1_dram_addr,
        .page_size = src1_tile_size,
        .data_format = src1_data_format
    };

    const InterleavedAddrGenFast<is_dram> s2 = {
        .bank_base_address = src2_dram_addr,
        .page_size = src2_tile_size,
        .data_format = src2_data_format
    };

    // load top and bottom tile first
    cb_reserve_back(cb_id_in0, single_tile);
    cb_reserve_back(cb_id_in1, single_tile);

    uint32_t top = get_read_ptr(cb_id_in0);
    uint32_t bottom = get_read_ptr(cb_id_in1);

    uint32_t *top_tile = (uint32_t *)top;
    noc_async_read_tile(0, s0, top);
    noc_async_read_barrier();
    noc_async_read_tile(0, s1, bottom);
    noc_async_read_barrier();
    
    cb_push_back(cb_id_in0, single_tile);
    cb_push_back(cb_id_in1, single_tile);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id_in2, single_tile);
        uint32_t output = get_read_ptr(cb_id_in2);
        uint32_t *output_tile = (uint32_t *)output;
        noc_async_read_tile(i, s2, output);
        noc_async_read_barrier();
        cb_push_back(cb_id_in2, single_tile);
    }
}