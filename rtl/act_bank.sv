// Activation memory bank.
//
// Layer inputs are stored N-way interleaved: element idx of an activation
// vector lives in bank (idx mod N) at address (idx div N). This lets the
// controller fetch N consecutive activations (one input tile) in a single
// cycle — one from each bank — while the requantize stage writes output
// block j's element (j*N + c) back into bank c at address j with no
// conflicts. Two parities of banks ping-pong between consecutive layers.
//
// Simple 1W/1R synchronous RAM, addressed by (batch index, element address).

`default_nettype none

module act_bank #(
    parameter int DATA_W = 8,
    parameter int B_MAX  = 16,
    parameter int DEPTH  = 200,           // >= ceil(784/4) elements per bank
    parameter int B_ADDR = $clog2(B_MAX),
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  wire                     clk,

    input  wire                     wen,
    input  wire  [B_ADDR-1:0]       wb,
    input  wire  [ADDR_W-1:0]       waddr,
    input  wire signed [DATA_W-1:0] wdata,

    input  wire  [B_ADDR-1:0]       rb,
    input  wire  [ADDR_W-1:0]       raddr,
    output logic signed [DATA_W-1:0] rdata  // 1-cycle read latency
);

    logic signed [DATA_W-1:0] mem [0:B_MAX*DEPTH-1];

    initial begin
        for (int k = 0; k < B_MAX*DEPTH; k++) mem[k] = '0;
    end

    always_ff @(posedge clk) begin
        if (wen)
            mem[wb*DEPTH + waddr] <= wdata;
        rdata <= mem[rb*DEPTH + raddr];
    end

endmodule

`default_nettype wire
