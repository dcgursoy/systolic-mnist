// Processing element for the weight-stationary systolic array.
//
// Holds one INT8 weight. Each cycle it consumes an activation from the west
// and a partial sum from the north, and registers:
//     psum_out <= psum_in + a_in * w      (INT8 x INT8 accumulated in INT32)
//     a_out    <= a_in                    (activation forwarded east)
//
// The weight is captured from w_in when w_load is asserted (the array's
// row-select decoder drives w_load for one row at a time during tile loads).

`default_nettype none

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  wire                       clk,
    input  wire                       rst_n,

    // weight load
    input  wire                       w_load,
    input  wire signed [DATA_W-1:0]   w_in,

    // systolic dataflow
    input  wire signed [DATA_W-1:0]   a_in,     // from west
    input  wire signed [ACC_W-1:0]    psum_in,  // from north
    output logic signed [DATA_W-1:0]  a_out,    // to east
    output logic signed [ACC_W-1:0]   psum_out, // to south

    // stored weight, exposed for tracing / debug
    output logic signed [DATA_W-1:0]  w_dbg
);

    logic signed [DATA_W-1:0] w_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_reg    <= '0;
            a_out    <= '0;
            psum_out <= '0;
        end else begin
            if (w_load)
                w_reg <= w_in;
            a_out    <= a_in;
            psum_out <= psum_in + a_in * w_reg;
        end
    end

    assign w_dbg = w_reg;

endmodule

`default_nettype wire
