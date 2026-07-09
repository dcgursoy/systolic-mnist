// Top level: weight-stationary systolic MLP accelerator.
//
// Executes the full quantized MLP (all layers) over a batch of images:
//
//   for each layer l:                        (config from accel_config.svh)
//     for each output block j:               (N output neurons)
//       for each input tile i:               (N input elements)
//         S_WLOAD : load the N x N weight tile, one row per cycle
//         S_STREAM: stream B activation vectors through the skewed array
//         S_DRAIN : per-column accumulators absorb B partial sums each
//       S_REQ     : requantize accumulators -> activation banks (hidden
//                   layers) or raw INT32 logits out (final layer)
//
// Weights/biases are pre-packed in exactly this loop order, so both memory
// pointers simply increment. Activations ping-pong between two parities of
// N interleaved banks so layer l's reads never collide with its writes.
//
// Batch dimension: streaming B vectors per tile amortizes the N-cycle
// weight load, which is the fundamental weight-stationary trade-off (see
// docs/architecture.md for the utilization analysis).

`default_nettype none

module accel_top #(
    parameter int N      = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    parameter int B_MAX  = 16
) (
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      start,
    input  wire [$clog2(B_MAX):0]    batch_size,   // 1 .. B_MAX

    // image load port (use while idle; writes parity-0 activation banks)
    input  wire                      img_wen,
    input  wire [$clog2(B_MAX)-1:0]  img_b,
    input  wire [9:0]                img_idx,      // 0 .. 783
    input  wire signed [DATA_W-1:0]  img_data,

    // result stream: one output block of raw INT32 logits per valid cycle
    output logic                     logits_valid,
    output wire signed [N*ACC_W-1:0] logits_flat,
    output logic [$clog2(B_MAX)-1:0] logits_b,     // batch index
    output logic [7:0]               logits_base,  // first output index (j*N)

    output logic                     done,
    output logic [31:0]              cycle_count
);

    `include "accel_config.svh"

    localparam int B_ADDR = $clog2(B_MAX);
    localparam int ROW_W  = $clog2(N);

    // ------------------------------------------------------------------
    // Controller state
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE, S_SETUP, S_WLOAD, S_STREAM, S_DRAIN, S_REQ, S_DONE
    } state_t;
    state_t state;

    logic [3:0]        layer;
    logic [7:0]        out_blk;
    logic [7:0]        in_tile;
    logic [15:0]       w_ptr;
    logic [7:0]        b_ptr;
    logic [ROW_W:0]    wl_cnt;
    logic [B_ADDR:0]   issue_b, drain_cnt, req_cnt;

    wire [15:0] n_tiles    = cfg_in_tiles(layer);
    wire [15:0] n_blocks   = cfg_out_blocks(layer);
    wire        last_layer = (layer == CFG_NUM_LAYERS - 1);
    wire        first_tile = (in_tile == 8'd0);

    // ------------------------------------------------------------------
    // Weight / bias memories (packed by sim/gen_config.py, synchronous read)
    // ------------------------------------------------------------------
    logic [N*DATA_W-1:0] wmem [0:CFG_W_WORDS-1];
    logic [N*ACC_W-1:0]  bmem [0:CFG_B_WORDS-1];
    logic [N*DATA_W-1:0] wdata_r;
    logic [N*ACC_W-1:0]  bias_word_r;

    always_ff @(posedge clk) begin
        wdata_r     <= wmem[w_ptr];
        bias_word_r <= bmem[b_ptr];
    end

    // ------------------------------------------------------------------
    // Activation banks: 2 parities x N interleaved banks.
    // Layer l reads parity l%2 and writes parity (l+1)%2.
    // ------------------------------------------------------------------
    logic                     req_wen_d;
    logic [B_ADDR-1:0]        req_b_d;
    wire signed [DATA_W-1:0]  req_y     [0:N-1];
    wire signed [DATA_W-1:0]  act_rdata [0:1][0:N-1];
    wire                      act_write   = req_wen_d && !last_layer;
    wire                      wr_parity   = ~layer[0];
    wire                      rd_parity   = layer[0];

    wire [$clog2(CFG_ACT_DEPTH)-1:0] img_addr = img_idx / N;

    genvar gp, gc;
    generate
        for (gp = 0; gp < 2; gp++) begin : g_parity
            for (gc = 0; gc < N; gc++) begin : g_bank
                wire img_hit = (gp == 0) && img_wen && (img_idx % N == gc);
                wire req_hit = act_write && (wr_parity == (gp != 0));
                act_bank #(
                    .DATA_W(DATA_W), .B_MAX(B_MAX), .DEPTH(CFG_ACT_DEPTH)
                ) u_act (
                    .clk   (clk),
                    .wen   (img_hit || req_hit),
                    .wb    (img_hit ? img_b : req_b_d),
                    .waddr (img_hit ? img_addr : out_blk[$clog2(CFG_ACT_DEPTH)-1:0]),
                    .wdata (img_hit ? img_data : req_y[gc]),
                    .rb    (issue_b[B_ADDR-1:0]),
                    .raddr (in_tile[$clog2(CFG_ACT_DEPTH)-1:0]),
                    .rdata (act_rdata[gp][gc])
                );
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Systolic array + input skew
    // ------------------------------------------------------------------
    wire signed [N*DATA_W-1:0] a_aligned;
    wire signed [N*DATA_W-1:0] a_skewed;
    wire signed [N*ACC_W-1:0]  psum_flat;

    generate
        for (gc = 0; gc < N; gc++) begin : g_avec
            assign a_aligned[gc*DATA_W +: DATA_W] =
                rd_parity ? act_rdata[1][gc] : act_rdata[0][gc];
        end
    endgenerate

    skew_buffer #(.N(N), .DATA_W(DATA_W)) u_skew (
        .clk(clk), .rst_n(rst_n), .a_flat(a_aligned), .a_skewed(a_skewed)
    );

    // wmem reads are registered, so the row-select trails the address
    // pointer by one cycle: wl_cnt=0 only issues the first read, rows are
    // written on wl_cnt = 1..N (S_WLOAD lasts N+1 cycles).
    wire [ROW_W:0] wl_row = wl_cnt - 1'b1;

    systolic_array #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .w_load_en  ((state == S_WLOAD) && (wl_cnt != 0)),
        .w_row_sel  (wl_row[ROW_W-1:0]),
        .w_col_flat (wdata_r),
        .a_flat     (a_skewed),
        .psum_flat  (psum_flat)
    );

    // Validity pipeline: a vector issued to the act banks appears at the
    // array's west edge one cycle later (registered bank read); column c's
    // partial sum reaches the bottom N + c cycles after that.
    logic vec_valid_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) vec_valid_d <= 1'b0;
        else        vec_valid_d <= (state == S_STREAM);
    end

    wire [N-1:0] drain_valid;
    generate
        for (gc = 0; gc < N; gc++) begin : g_vchain
            logic [N+gc-1:0] sh;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) sh <= '0;
                else        sh <= {sh[N+gc-2:0], vec_valid_d};
            end
            assign drain_valid[gc] = sh[N+gc-1];
        end
    endgenerate

    // ------------------------------------------------------------------
    // Per-column accumulators and requantize units
    // ------------------------------------------------------------------
    wire signed [ACC_W-1:0] acc_rdata [0:N-1];
    wire tile_start = (state == S_WLOAD) && (wl_cnt == '0);

    generate
        for (gc = 0; gc < N; gc++) begin : g_acc
            acc_bank #(.ACC_W(ACC_W), .B_MAX(B_MAX)) u_acc (
                .clk         (clk),
                .rst_n       (rst_n),
                .tile_start  (tile_start),
                .first_tile  (first_tile),
                .bias        (bias_word_r[gc*ACC_W +: ACC_W]),
                .drain_valid (drain_valid[gc]),
                .psum        (psum_flat[gc*ACC_W +: ACC_W]),
                .raddr       (req_cnt[B_ADDR-1:0]),
                .rdata       (acc_rdata[gc])
            );
            requantize #(.ACC_W(ACC_W), .M_W(16), .DATA_W(DATA_W)) u_req (
                .acc   (acc_rdata[gc]),
                .m     (cfg_m(layer)),
                .shift (cfg_shift(layer)),
                .y     (req_y[gc])
            );
            assign logits_flat[gc*ACC_W +: ACC_W] = acc_rdata[gc];
        end
    endgenerate

    assign logits_valid = req_wen_d && last_layer;

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            layer       <= '0;
            out_blk     <= '0;
            in_tile     <= '0;
            w_ptr       <= '0;
            b_ptr       <= '0;
            wl_cnt      <= '0;
            issue_b     <= '0;
            drain_cnt   <= '0;
            req_cnt     <= '0;
            req_wen_d   <= 1'b0;
            req_b_d     <= '0;
            logits_b    <= '0;
            logits_base <= '0;
            done        <= 1'b0;
            cycle_count <= '0;
        end else begin
            req_wen_d <= (state == S_REQ) && (req_cnt < batch_size);
            req_b_d   <= req_cnt[B_ADDR-1:0];
            if (state == S_REQ) begin
                logits_b    <= req_cnt[B_ADDR-1:0];
                logits_base <= out_blk * N;
            end
            if (state != S_IDLE && state != S_DONE)
                cycle_count <= cycle_count + 1;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        layer       <= '0;
                        out_blk     <= '0;
                        in_tile     <= '0;
                        w_ptr       <= '0;
                        b_ptr       <= '0;
                        done        <= 1'b0;
                        cycle_count <= '0;
                        state       <= S_SETUP;
                    end
                end

                // one dead cycle so wdata_r/bias_word_r reflect the pointers
                S_SETUP: begin
                    wl_cnt <= '0;
                    state  <= S_WLOAD;
                end

                S_WLOAD: begin
                    if (wl_cnt < N)
                        w_ptr <= w_ptr + 1;
                    wl_cnt    <= wl_cnt + 1;
                    drain_cnt <= '0;
                    if (wl_cnt == N) begin
                        issue_b <= '0;
                        state   <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    issue_b <= issue_b + 1;
                    if (drain_valid[N-1])
                        drain_cnt <= drain_cnt + 1;
                    if (issue_b == batch_size - 1)
                        state <= S_DRAIN;
                end

                S_DRAIN: begin
                    if (drain_valid[N-1])
                        drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == batch_size) begin
                        if (in_tile == n_tiles - 1) begin
                            req_cnt <= '0;
                            state   <= S_REQ;
                        end else begin
                            in_tile <= in_tile + 1;
                            wl_cnt  <= '0;
                            state   <= S_WLOAD;
                        end
                    end
                end

                S_REQ: begin
                    req_cnt <= req_cnt + 1;
                    if (req_cnt == batch_size) begin
                        b_ptr   <= b_ptr + 1;
                        in_tile <= '0;
                        wl_cnt  <= '0;
                        if (out_blk == n_blocks - 1) begin
                            if (last_layer) begin
                                done  <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                layer   <= layer + 1;
                                out_blk <= '0;
                                state   <= S_WLOAD;
                            end
                        end else begin
                            out_blk <= out_blk + 1;
                            state   <= S_WLOAD;
                        end
                    end
                end

                S_DONE: state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
