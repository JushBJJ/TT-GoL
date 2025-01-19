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

    DPRINT << "(READER): About to start kernel_main" << ENDL();

    // load top and bottom tile first
    DPRINT << "(READER): Waiting for cb_id_in0 and cb_id_in1" << ENDL();
    cb_reserve_back(cb_id_in0, single_tile);
    cb_reserve_back(cb_id_in1, single_tile);

    DPRINT << "(READER): Getting read pointers for cb_id_in0 and cb_id_in1" << ENDL();
    uint32_t top = get_read_ptr(cb_id_in0);
    uint32_t bottom = get_read_ptr(cb_id_in1);

    DPRINT << "(READER): Writing to NoC" << ENDL();

    uint32_t *top_tile = (uint32_t *)top;

    DPRINT << "(READER) top_tile[0] = " << top_tile[0] << ENDL();

    noc_async_read_tile(0, s0, top);
    noc_async_read_barrier();
    noc_async_read_tile(0, s1, bottom);
    noc_async_read_barrier();
    
    DPRINT << "(READER): Popping front for cb_id_in0 and cb_id_in1" << ENDL();

    cb_push_back(cb_id_in0, single_tile);
    cb_push_back(cb_id_in1, single_tile);

    DPRINT << "(READER): Reading from src0" << ENDL();


    for (uint32_t i = 0; i < num_tiles; i++) {
        DPRINT << "(READER) Iteration " << i << ": Waiting for cb_id_in2" << ENDL();
        cb_reserve_back(cb_id_in2, single_tile);

        DPRINT << "(READER) Iteration " << i << ": Getting read pointer for cb_id_in2" << ENDL();
        uint32_t output = get_read_ptr(cb_id_in2);
        uint32_t *output_tile = (uint32_t *)output;

        DPRINT << "(READER) Iteration " << i << ": Writing tile to NoC" << ENDL();
        noc_async_read_tile(i, s2, output);
        noc_async_read_barrier();

        DPRINT << "(READER) Iteration " << i << ": Popping front for cb_id_in2" << ENDL();
        cb_push_back(cb_id_in2, single_tile);
    }


    DPRINT << "(READER): Finished kernel_main" << ENDL();
}