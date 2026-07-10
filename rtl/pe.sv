// Processing element for the weight-stationary systolic array,
// with double-buffered weights.
//
// Two weight registers: w_shadow is loaded through the load port (row by
// row) while the array is busy computing with w_active. A swap token
// (swap_in) travels west→east with the same one-cycle-per-PE delay as the
// activations; when it passes, the PE promotes shadow → active. Because the
// token moves on the same wavefront as the data, the next tile's
// activations can follow exactly one cycle behind it — weight reloads cost
// zero pipeline bubbles.
//
// Per cycle:
//     psum_out <= psum_in + a_in * w_active   (INT8 x INT8 into INT32)
//     a_out    <= a_in
//     swap_out <= swap_in

`default_nettype none

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // shadow-weight load
    input  wire                       w_load,
    input  wire signed [DATA_W-1:0]   w_in,

    // systolic dataflow
    input  wire signed [DATA_W-1:0]   a_in,     // from west
    input  wire signed [ACC_W-1:0]    psum_in,  // from north
    input  wire                       swap_in,  // from west
    output logic signed [DATA_W-1:0]  a_out,    // to east
    output logic signed [ACC_W-1:0]   psum_out, // to south
    output logic                      swap_out, // to east

    // active weight, exposed for tracing / debug
    output logic signed [DATA_W-1:0]  w_dbg
);

    logic signed [DATA_W-1:0] w_shadow, w_active;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_shadow <= '0;
            w_active <= '0;
            a_out    <= '0;
            psum_out <= '0;
            swap_out <= 1'b0;
        end else begin
            if (w_load)
                w_shadow <= w_in;
            if (swap_in)
                w_active <= w_shadow;
            swap_out <= swap_in;
            a_out    <= a_in;
            psum_out <= psum_in + a_in * w_active;
        end
    end

    assign w_dbg = w_active;

endmodule

`default_nettype wire
