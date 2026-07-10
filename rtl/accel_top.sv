// Top level: weight-stationary systolic MLP accelerator with
// double-buffered weight loading and a pipelined requantizer.
//
// Executes the full quantized MLP (all layers) over a batch of images:
//
//   for each layer l:                        (config from accel_config.svh)
//     for each output block j:               (N output neurons)
//       S_PRELOAD: load tile 0 into the PEs' shadow registers
//       S_RUN, for each input tile i:
//         slot 0   : issue the swap token (promotes shadow -> active as it
//                    sweeps the array on the data wavefront)
//         slots 1..: stream B activation vectors, zero bubble behind the
//                    token; from slot N, load tile i+1's shadow registers
//                    underneath the computation
//       S_TAIL   : let the last tile's partial sums drain
//       S_REQ    : requantize accumulators -> activation banks (hidden
//                  layers) or raw INT32 logits out (final layer)
//
// The swap-token timing is the point: reloading weights costs no array
// idle cycles except the shadow-load *latency floor* (visible only when
// batch_size < N+2). Tile i's results drain while tile i+1 streams, so the
// per-column valid pipelines also carry a `first-tile` flag telling the
// accumulator banks when to inject the bias.
//
// Weights/biases are pre-packed in exactly this loop order, so both memory
// pointers simply increment. Activations ping-pong between two parities of
// N interleaved banks so layer l's reads never collide with its writes.

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
    output wire  [$clog2(B_MAX)-1:0] logits_b,     // batch index
    output wire  [7:0]               logits_base,  // first output index (j*N)

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
        S_IDLE, S_SETUP, S_PRELOAD, S_RUN, S_TAIL, S_REQ, S_DONE
    } state_t;
    state_t state;

    logic [3:0]        layer;
    logic [7:0]        out_blk;
    logic [7:0]        in_tile;      // tile currently being issued
    logic [15:0]       w_ptr;
    logic [7:0]        b_ptr;
    logic [7:0]        slot;         // cycle slot within the current tile
    logic [B_ADDR:0]   issue_b;
    logic [12:0]       block_drain_cnt, target_drains;
    logic [B_ADDR+1:0] req_cnt;

    // shadow-load engine (runs during S_PRELOAD and underneath S_RUN)
    logic              wl_run;
    logic [ROW_W:0]    wl_cnt;
    logic              loaded_next;  // next tile's shadow is fully loaded

    wire [15:0] n_tiles    = cfg_in_tiles(layer);
    wire [15:0] n_blocks   = cfg_out_blocks(layer);
    wire        last_layer = (layer == CFG_NUM_LAYERS - 1);
    wire        last_tile  = (in_tile == n_tiles - 1);

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
    // Requantize write pipeline (2 stages behind the acc-bank read):
    //   cycle k   : raddr = req_cnt
    //   cycle k+1 : acc_rdata valid; requantize stage-1 registers acc*M
    //   cycle k+2 : y valid; write activations / emit logits
    // ------------------------------------------------------------------
    logic                     req_wen_d, req_wen_d2;
    logic [B_ADDR-1:0]        req_b_d, req_b_d2;
    logic signed [ACC_W-1:0]  acc_rdata_d [0:N-1];

    // ------------------------------------------------------------------
    // Activation banks: 2 parities x N interleaved banks.
    // Layer l reads parity l%2 and writes parity (l+1)%2.
    // ------------------------------------------------------------------
    wire signed [DATA_W-1:0]  req_y     [0:N-1];
    wire signed [DATA_W-1:0]  act_rdata [0:1][0:N-1];
    wire                      act_write   = req_wen_d2 && !last_layer;
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
                    .wb    (img_hit ? img_b : req_b_d2),
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
    // Systolic array + input skew (activations and the swap token share
    // the same per-row skew so the token rides the data wavefront)
    // ------------------------------------------------------------------
    wire signed [N*DATA_W-1:0] a_aligned;
    wire signed [N*DATA_W-1:0] a_skewed;
    wire        [N-1:0]        swap_skewed;
    wire signed [N*ACC_W-1:0]  psum_flat;
    logic                      vec_valid_d, first_d, token_d;

    generate
        for (gc = 0; gc < N; gc++) begin : g_avec
            assign a_aligned[gc*DATA_W +: DATA_W] =
                rd_parity ? act_rdata[1][gc] : act_rdata[0][gc];
        end
    endgenerate

    skew_buffer #(.N(N), .DATA_W(DATA_W)) u_skew (
        .clk(clk), .rst_n(rst_n), .a_flat(a_aligned), .a_skewed(a_skewed)
    );

    skew_buffer #(.N(N), .DATA_W(1)) u_swap_skew (
        .clk(clk), .rst_n(rst_n), .a_flat({N{token_d}}), .a_skewed(swap_skewed)
    );

    systolic_array #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .w_load_en  (wl_run && (wl_cnt != 0)),
        .w_row_sel  (wl_cnt[ROW_W-1:0] - 1'b1),
        .w_col_flat (wdata_r),
        .a_flat     (a_skewed),
        .swap_flat  (swap_skewed),
        .psum_flat  (psum_flat)
    );

    // Validity pipeline: a vector issued to the act banks appears at the
    // array's west edge one cycle later (registered bank read); column c's
    // partial sum reaches the bottom N + c cycles after that. A parallel
    // bit marks drains belonging to tile 0 of the block (bias injection).
    wire [N-1:0] drain_valid, drain_first;
    generate
        for (gc = 0; gc < N; gc++) begin : g_vchain
            logic [N+gc-1:0] sh_v, sh_f;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sh_v <= '0;
                    sh_f <= '0;
                end else begin
                    sh_v <= {sh_v[N+gc-2:0], vec_valid_d};
                    sh_f <= {sh_f[N+gc-2:0], first_d};
                end
            end
            assign drain_valid[gc] = sh_v[N+gc-1];
            assign drain_first[gc] = sh_f[N+gc-1];
        end
    endgenerate

    // ------------------------------------------------------------------
    // Per-column accumulators and requantize units
    // ------------------------------------------------------------------
    wire signed [ACC_W-1:0] acc_rdata [0:N-1];
    wire block_start = (state == S_PRELOAD) && (wl_cnt == '0);

    generate
        for (gc = 0; gc < N; gc++) begin : g_acc
            acc_bank #(.ACC_W(ACC_W), .B_MAX(B_MAX)) u_acc (
                .clk         (clk),
                .rst_n       (rst_n),
                .block_start (block_start),
                .batch_size  (batch_size),
                .bias        (bias_word_r[gc*ACC_W +: ACC_W]),
                .drain_valid (drain_valid[gc]),
                .first       (drain_first[gc]),
                .psum        (psum_flat[gc*ACC_W +: ACC_W]),
                .raddr       (req_cnt[B_ADDR-1:0]),
                .rdata       (acc_rdata[gc])
            );
            requantize #(.ACC_W(ACC_W), .M_W(16), .DATA_W(DATA_W)) u_req (
                .clk   (clk),
                .acc   (acc_rdata[gc]),
                .m     (cfg_m(layer)),
                .shift (cfg_shift(layer)),
                .y     (req_y[gc])
            );
            assign logits_flat[gc*ACC_W +: ACC_W] = acc_rdata_d[gc];
        end
    endgenerate

    assign logits_valid = req_wen_d2 && last_layer;
    assign logits_b     = req_b_d2;
    assign logits_base  = out_blk * N;

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            layer           <= '0;
            out_blk         <= '0;
            in_tile         <= '0;
            w_ptr           <= '0;
            b_ptr           <= '0;
            slot            <= '0;
            issue_b         <= '0;
            block_drain_cnt <= '0;
            target_drains   <= '0;
            req_cnt         <= '0;
            wl_run          <= 1'b0;
            wl_cnt          <= '0;
            loaded_next     <= 1'b0;
            vec_valid_d     <= 1'b0;
            first_d         <= 1'b0;
            token_d         <= 1'b0;
            req_wen_d       <= 1'b0;
            req_wen_d2      <= 1'b0;
            req_b_d         <= '0;
            req_b_d2        <= '0;
            done            <= 1'b0;
            cycle_count     <= '0;
            for (i = 0; i < N; i = i + 1) acc_rdata_d[i] <= '0;
        end else begin
            // stream-issue pipeline flags
            vec_valid_d <= (state == S_RUN) && (slot >= 1)
                           && (issue_b < batch_size);
            first_d     <= (state == S_RUN) && (slot >= 1)
                           && (issue_b < batch_size) && (in_tile == 0);
            token_d     <= (state == S_RUN) && (slot == 0);

            // requantize write pipeline
            req_wen_d  <= (state == S_REQ) && (req_cnt < batch_size);
            req_b_d    <= req_cnt[B_ADDR-1:0];
            req_wen_d2 <= req_wen_d;
            req_b_d2   <= req_b_d;
            for (i = 0; i < N; i = i + 1) acc_rdata_d[i] <= acc_rdata[i];

            // shadow-load engine
            if (wl_run) begin
                if (wl_cnt < N)
                    w_ptr <= w_ptr + 1;
                wl_cnt <= wl_cnt + 1;
                if (wl_cnt == N) begin
                    wl_run      <= 1'b0;
                    loaded_next <= 1'b1;
                end
            end

            // drain bookkeeping (tiles drain across S_RUN and S_TAIL)
            if (drain_valid[N-1])
                block_drain_cnt <= block_drain_cnt + 1;

            if (state != S_IDLE && state != S_DONE)
                cycle_count <= cycle_count + 1;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        layer       <= '0;
                        out_blk     <= '0;
                        w_ptr       <= '0;
                        b_ptr       <= '0;
                        done        <= 1'b0;
                        cycle_count <= '0;
                        state       <= S_SETUP;
                    end
                end

                // one dead cycle so wdata_r/bias_word_r reflect the pointers
                S_SETUP: begin
                    wl_run          <= 1'b1;
                    wl_cnt          <= '0;
                    loaded_next     <= 1'b0;
                    block_drain_cnt <= '0;
                    target_drains   <= n_tiles * batch_size;
                    state           <= S_PRELOAD;
                end

                S_PRELOAD: begin
                    if (wl_cnt == N) begin   // final shadow row this cycle
                        in_tile     <= '0;
                        slot        <= '0;
                        issue_b     <= '0;
                        loaded_next <= 1'b0;
                        state       <= S_RUN;
                    end
                end

                S_RUN: begin
                    slot <= slot + 1;
                    if ((slot >= 1) && (issue_b < batch_size))
                        issue_b <= issue_b + 1;

                    // load next tile's shadow beneath the stream; slot N is
                    // the earliest write that cannot race the swap token
                    if ((slot == N - 1) && !last_tile
                        && !wl_run && !loaded_next) begin
                        wl_run <= 1'b1;
                        wl_cnt <= '0;
                    end

                    if (issue_b == batch_size) begin
                        if (last_tile)
                            state <= S_TAIL;
                        else if (loaded_next) begin
                            in_tile     <= in_tile + 1;
                            slot        <= '0;
                            issue_b     <= '0;
                            loaded_next <= 1'b0;
                        end
                    end
                end

                S_TAIL: begin
                    if (block_drain_cnt == target_drains) begin
                        req_cnt <= '0;
                        state   <= S_REQ;
                    end
                end

                S_REQ: begin
                    req_cnt <= req_cnt + 1;
                    if (req_cnt == batch_size + 1) begin
                        b_ptr <= b_ptr + 1;
                        if (out_blk == n_blocks - 1) begin
                            if (last_layer) begin
                                done  <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                layer   <= layer + 1;
                                out_blk <= '0;
                                state   <= S_SETUP;
                            end
                        end else begin
                            out_blk <= out_blk + 1;
                            state   <= S_SETUP;
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
