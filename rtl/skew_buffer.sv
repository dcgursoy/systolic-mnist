// Input skew buffer.
//
// A weight-stationary array needs activation row r to enter the west edge
// r cycles after row 0 so that partial sums cascading down each column meet
// the right activation at the right time. This module delays row r by r
// cycles (row 0 passes through combinationally), producing the diagonal
// "systolic wave" visible in the trace.

`default_nettype none

module skew_buffer #(
    parameter int N      = 4,
    parameter int DATA_W = 8
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire signed [N*DATA_W-1:0]  a_flat,   // aligned activation vector
    output wire signed [N*DATA_W-1:0]  a_skewed  // row r delayed by r cycles
);

    genvar r;
    generate
        for (r = 0; r < N; r++) begin : g_row
            if (r == 0) begin : g_pass
                assign a_skewed[DATA_W-1:0] = a_flat[DATA_W-1:0];
            end else begin : g_delay
                logic signed [DATA_W-1:0] pipe [0:r-1];
                integer k;
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (k = 0; k < r; k = k + 1)
                            pipe[k] <= '0;
                    end else begin
                        pipe[0] <= a_flat[r*DATA_W +: DATA_W];
                        for (k = 1; k < r; k = k + 1)
                            pipe[k] <= pipe[k-1];
                    end
                end
                assign a_skewed[r*DATA_W +: DATA_W] = pipe[r-1];
            end
        end
    endgenerate

endmodule

`default_nettype wire
