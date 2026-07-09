// Per-column partial-sum accumulator bank.
//
// One instance sits under each array column. While a weight tile's results
// drain out of the column (one INT32 partial sum per streamed batch vector),
// the bank adds each into its per-image accumulator:
//
//     acc[b] <= (first_tile ? bias : acc[b]) + psum
//
// Injecting the bias on the first input tile of an output block means no
// separate accumulator-initialization pass is needed. An internal counter
// tracks which batch index the incoming drain belongs to (drains arrive in
// batch order); tile_start resets it.

`default_nettype none

module acc_bank #(
    parameter int ACC_W  = 32,
    parameter int B_MAX  = 16,
    parameter int B_ADDR = $clog2(B_MAX)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     tile_start,   // reset batch counter
    input  wire                     first_tile,   // this tile is in_tile 0
    input  wire signed [ACC_W-1:0]  bias,

    input  wire                     drain_valid,
    input  wire signed [ACC_W-1:0]  psum,

    input  wire  [B_ADDR-1:0]       raddr,
    output logic signed [ACC_W-1:0] rdata         // 1-cycle read latency
);

    logic signed [ACC_W-1:0] mem [0:B_MAX-1];
    logic [B_ADDR-1:0] wcnt;

    // keep simulation (and the JSON trace) free of X before first write
    initial begin
        for (int k = 0; k < B_MAX; k++) mem[k] = '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wcnt <= '0;
        end else if (tile_start) begin
            wcnt <= '0;
        end else if (drain_valid) begin
            mem[wcnt] <= (first_tile ? bias : mem[wcnt]) + psum;
            wcnt      <= wcnt + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        rdata <= mem[raddr];
    end

endmodule

`default_nettype wire
