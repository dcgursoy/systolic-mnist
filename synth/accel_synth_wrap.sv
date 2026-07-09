// Synthesis wrapper: serializes accel_top's wide I/O so the design fits a
// real FPGA pinout for the place-and-route timing study. Inputs deserialize
// from a 1-bit shift register; outputs XOR-reduce into one registered bit.
// This keeps every internal path intact (nothing optimizes away — the XOR
// cone observes all outputs) while using only 4 package pins.

`default_nettype none

module accel_synth_wrap #(
    parameter int N = 4
) (
    input  wire clk,
    input  wire rst_n,
    input  wire si,
    output logic so
);

    localparam int B_MAX = 16;
    localparam int B_ADDR = $clog2(B_MAX);
    localparam int IN_W = 1 + (B_ADDR+1) + 1 + B_ADDR + 10 + 8;

    logic [IN_W-1:0] in_sr;
    always_ff @(posedge clk) in_sr <= {in_sr[IN_W-2:0], si};

    wire                    logits_valid;
    wire [N*32-1:0]         logits_flat;
    wire [B_ADDR-1:0]       logits_b;
    wire [7:0]              logits_base;
    wire                    done;
    wire [31:0]             cycle_count;

    accel_top #(.N(N), .B_MAX(B_MAX)) u_core (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (in_sr[0]),
        .batch_size  (in_sr[B_ADDR+1:1]),
        .img_wen     (in_sr[B_ADDR+2]),
        .img_b       (in_sr[2*B_ADDR+2:B_ADDR+3]),
        .img_idx     (in_sr[2*B_ADDR+12:2*B_ADDR+3]),
        .img_data    (in_sr[2*B_ADDR+20:2*B_ADDR+13]),
        .logits_valid(logits_valid),
        .logits_flat (logits_flat),
        .logits_b    (logits_b),
        .logits_base (logits_base),
        .done        (done),
        .cycle_count (cycle_count)
    );

    always_ff @(posedge clk)
        so <= ^{logits_valid, logits_flat, logits_b, logits_base,
                done, cycle_count};

endmodule

`default_nettype wire
