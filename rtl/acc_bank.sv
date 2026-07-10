// Per-column partial-sum accumulator bank.
//
// One instance sits under each array column. Tiles drain back-to-back with
// weight double-buffering, so the bank cycles its batch counter modulo
// batch_size (each tile contributes exactly batch_size drains, in batch
// order) instead of being reset per tile. The bias is injected on drains
// belonging to the first tile of an output block — that information rides
// down the controller's per-column valid pipeline as the `first` bit,
// because by the time tile 0's results drain, the controller is already
// issuing tile 1.
//
//     acc[b] <= (first ? bias : acc[b]) + psum

`default_nettype none

module acc_bank #(
    parameter int ACC_W  = 32,
    parameter int B_MAX  = 16,
    parameter int B_ADDR = $clog2(B_MAX)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     block_start,  // reset batch counter
    input  wire  [B_ADDR:0]         batch_size,
    input  wire signed [ACC_W-1:0]  bias,

    input  wire                     drain_valid,
    input  wire                     first,        // drain is from tile 0
    input  wire signed [ACC_W-1:0]  psum,

    input  wire  [B_ADDR-1:0]       raddr,
    output logic signed [ACC_W-1:0] rdata         // 1-cycle read latency
);

    logic signed [ACC_W-1:0] mem [0:B_MAX-1];
    logic [B_ADDR:0] wcnt;

    // keep simulation (and the JSON trace) free of X before first write
    initial begin
        for (int k = 0; k < B_MAX; k++) mem[k] = '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wcnt <= '0;
        end else if (block_start) begin
            wcnt <= '0;
        end else if (drain_valid) begin
            mem[wcnt[B_ADDR-1:0]] <= (first ? bias : mem[wcnt[B_ADDR-1:0]]) + psum;
            wcnt <= (wcnt == batch_size - 1) ? '0 : wcnt + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        rdata <= mem[raddr];
    end

endmodule

`default_nettype wire
