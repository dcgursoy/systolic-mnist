// N x N weight-stationary systolic array.
//
// Mapping for y = W_tile @ x_tile (a 4x4 sub-block of a layer's matmul):
//   - PE(r,c) holds weight W[out = c][in = r], i.e. column c owns output c.
//   - Activation x[r] enters row r at the west edge (pre-skewed by
//     skew_buffer) and propagates east, one PE per cycle.
//   - Partial sums propagate south; column c's bottom output is
//     sum_r W[c][r] * x[r], emerging N + c cycles after x entered.
//
// Weight loading: w_col_flat carries one weight per column; w_row_sel picks
// which row captures it. A full tile loads in N cycles (one row per cycle).
//
// The inter-PE nets (a_h, psum_v) and stored weights (w_dbg) are module-level
// arrays so the testbench can index them at runtime to dump the per-cycle
// trace that drives the visualizer.

`default_nettype none

module systolic_array #(
    parameter int N      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  wire                        clk,
    input  wire                        rst_n,

    // weight load port (one row per cycle)
    input  wire                        w_load_en,
    input  wire  [$clog2(N)-1:0]       w_row_sel,
    input  wire signed [N*DATA_W-1:0]  w_col_flat,

    // activation stream (already skewed), one lane per row
    input  wire signed [N*DATA_W-1:0]  a_flat,

    // bottom-edge partial sums, one lane per column (column c skewed +c)
    output wire signed [N*ACC_W-1:0]   psum_flat
);

    // Inter-PE nets. a_h[r][c] feeds PE(r,c) from the west;
    // psum_v[r][c] feeds PE(r,c) from the north.
    wire signed [DATA_W-1:0] a_h    [0:N-1][0:N];
    wire signed [ACC_W-1:0]  psum_v [0:N]  [0:N-1];
    wire signed [DATA_W-1:0] w_dbg  [0:N-1][0:N-1];

    genvar r, c;
    generate
        for (r = 0; r < N; r++) begin : g_row
            // west edge: skewed activations; north edge: zero partial sums
            assign a_h[r][0] = a_flat[r*DATA_W +: DATA_W];
        end
        for (c = 0; c < N; c++) begin : g_col
            assign psum_v[0][c] = '0;
        end

        for (r = 0; r < N; r++) begin : g_pe_row
            for (c = 0; c < N; c++) begin : g_pe_col
                pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .w_load   (w_load_en && (w_row_sel == r)),
                    .w_in     (w_col_flat[c*DATA_W +: DATA_W]),
                    .a_in     (a_h[r][c]),
                    .psum_in  (psum_v[r][c]),
                    .a_out    (a_h[r][c+1]),
                    .psum_out (psum_v[r+1][c]),
                    .w_dbg    (w_dbg[r][c])
                );
            end
        end

        for (c = 0; c < N; c++) begin : g_out
            assign psum_flat[c*ACC_W +: ACC_W] = psum_v[N][c];
        end
    endgenerate

endmodule

`default_nettype wire
