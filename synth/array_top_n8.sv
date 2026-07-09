// Synthesis-only wrapper: instantiates the systolic array at N=8 as a
// default parameter. Exists because `hierarchy -chparam` asserts in the
// bundled yosys build when re-parameterizing a module.

`default_nettype none

module array_top_n8 #(
    parameter int N = 8,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        w_load_en,
    input  wire  [$clog2(N)-1:0]       w_row_sel,
    input  wire signed [N*DATA_W-1:0]  w_col_flat,
    input  wire signed [N*DATA_W-1:0]  a_flat,
    output wire signed [N*ACC_W-1:0]   psum_flat
);

    systolic_array #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (.*);

endmodule

`default_nettype wire
