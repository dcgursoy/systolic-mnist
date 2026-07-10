// Requantization unit: INT32 accumulator -> INT8 activation. Pipelined.
//
// Implements the gemmlowp/TFLite-style integer rescale specified in
// model/golden.py (the bit-exact contract):
//
//     y = clamp( (acc * M + 2^(shift-1)) >>> shift , 0, 127 )
//
// M is a per-layer int16 multiplier normalized into [2^14, 2^15); shift is a
// per-layer constant. The arithmetic shift with additive bias implements
// round-half-up; clamping the low side at 0 folds in the ReLU.
//
// Stage 1 (registered): the 32x16 multiply + rounding-bias add — the
// longest arithmetic path in the design, which is why it gets the register.
// Stage 2 (combinational): arithmetic shift + clamp. y is valid one cycle
// after acc.

`default_nettype none

module requantize #(
    parameter int ACC_W  = 32,
    parameter int M_W    = 16,
    parameter int DATA_W = 8
) (
    input  wire                      clk,
    input  wire signed [ACC_W-1:0]   acc,
    input  wire signed [M_W-1:0]     m,
    input  wire        [4:0]         shift,
    output logic signed [DATA_W-1:0] y
);

    // rounding constant must be a signed operand: a concatenation is
    // unsigned and would force the whole sum (incl. acc*m) unsigned,
    // zero-extending negative accumulators
    localparam logic signed [ACC_W+M_W-1:0] ONE = 1;

    logic signed [ACC_W+M_W-1:0] prod_q;
    logic signed [ACC_W+M_W-1:0] shifted;

    always_ff @(posedge clk)
        prod_q <= acc * m + (ONE <<< (shift - 5'd1));

    always_comb begin
        shifted = prod_q >>> shift;
        if (shifted < 0)
            y = '0;                        // ReLU folded into low clamp
        else if (shifted > 127)
            y = 8'sd127;
        else
            y = shifted[DATA_W-1:0];
    end

endmodule

`default_nettype wire
