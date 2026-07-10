// Smoke test: bare skew_buffer + systolic_array.
//
// Loads a known 4x4 weight tile, streams a few activation vectors, and
// checks each column's drained partial sum against a reference dot product
// computed in the testbench. Verifies the core dataflow + drain timing
// (column c valid N + c cycles after a vector enters the west edge).

`timescale 1ns / 1ps

module tb_array_smoke;

    localparam int N = 4;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;
    localparam int NVEC = 3;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                      w_load_en = 0;
    logic [$clog2(N)-1:0]      w_row_sel = 0;
    logic signed [N*DATA_W-1:0] w_col_flat = '0;
    logic signed [N*DATA_W-1:0] a_aligned = '0;
    logic                      token = 0;
    wire  signed [N*DATA_W-1:0] a_skewed;
    wire  [N-1:0]              swap_skewed;
    wire  signed [N*ACC_W-1:0]  psum_flat;

    skew_buffer #(.N(N), .DATA_W(DATA_W)) u_skew (
        .clk(clk), .rst_n(rst_n), .a_flat(a_aligned), .a_skewed(a_skewed));

    skew_buffer #(.N(N), .DATA_W(1)) u_tok_skew (
        .clk(clk), .rst_n(rst_n), .a_flat({N{token}}), .a_skewed(swap_skewed));

    systolic_array #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
        .clk(clk), .rst_n(rst_n),
        .w_load_en(w_load_en), .w_row_sel(w_row_sel), .w_col_flat(w_col_flat),
        .a_flat(a_skewed), .swap_flat(swap_skewed), .psum_flat(psum_flat));

    // W[r][c]: weight held by PE(r,c); column c computes sum_r W[r][c]*x[r]
    int W [0:N-1][0:N-1];
    int X [0:NVEC-1][0:N-1];
    int expected [0:NVEC-1][0:N-1];
    int errors = 0;

    // valid chain replica: drain_valid[c] = vec_valid delayed N+c cycles
    logic vec_valid = 0;
    logic [2*N-1:0] vch [0:N-1];
    wire  [N-1:0] drain_valid;
    genvar gc;
    generate
        for (gc = 0; gc < N; gc++) begin : g_v
            always_ff @(posedge clk) vch[gc] <= {vch[gc][2*N-2:0], vec_valid};
            assign drain_valid[gc] = vch[gc][N+gc-1];
        end
    endgenerate

    // per-column capture of drained results
    int vec_idx [0:N-1];
    integer c2;
    always @(posedge clk) begin
        for (c2 = 0; c2 < N; c2 = c2 + 1) begin
            if (drain_valid[c2]) begin
                if (psum_flat[c2*ACC_W +: ACC_W] !== expected[vec_idx[c2]][c2]) begin
                    $display("FAIL: vec %0d col %0d got %0d expected %0d",
                             vec_idx[c2], c2,
                             $signed(psum_flat[c2*ACC_W +: ACC_W]),
                             expected[vec_idx[c2]][c2]);
                    errors = errors + 1;
                end
                vec_idx[c2] = vec_idx[c2] + 1;
            end
        end
    end

    integer r, c, v;
    initial begin
        // test data: mixed signs, includes zeros
        for (r = 0; r < N; r++)
            for (c = 0; c < N; c++)
                W[r][c] = ((r * N + c + 1) * ((r + c) % 3 == 0 ? -1 : 1)) % 128;
        X[0][0] = 1;    X[0][1] = 2;   X[0][2] = 3;   X[0][3] = 4;
        X[1][0] = -5;   X[1][1] = 0;   X[1][2] = 127; X[1][3] = -128;
        X[2][0] = 0;    X[2][1] = 0;   X[2][2] = 0;   X[2][3] = 0;
        for (v = 0; v < NVEC; v++)
            for (c = 0; c < N; c++) begin
                expected[v][c] = 0;
                for (r = 0; r < N; r++)
                    expected[v][c] = expected[v][c] + W[r][c] * X[v][r];
            end
        for (c = 0; c < N; c++) vec_idx[c] = 0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // load weights into the shadow registers, one row per cycle
        for (r = 0; r < N; r++) begin
            w_load_en <= 1;
            w_row_sel <= r[$clog2(N)-1:0];
            for (c = 0; c < N; c++)
                w_col_flat[c*DATA_W +: DATA_W] <= W[r][c][DATA_W-1:0];
            @(posedge clk);
        end
        w_load_en <= 0;

        // swap token promotes shadow -> active on the data wavefront;
        // vectors follow one cycle behind
        token <= 1;
        @(posedge clk);
        token <= 0;

        // stream vectors back-to-back
        for (v = 0; v < NVEC; v++) begin
            for (r = 0; r < N; r++)
                a_aligned[r*DATA_W +: DATA_W] <= X[v][r][DATA_W-1:0];
            vec_valid <= 1;
            @(posedge clk);
        end
        vec_valid <= 0;
        a_aligned <= '0;

        repeat (3 * N) @(posedge clk);

        if (errors == 0 && vec_idx[N-1] == NVEC)
            $display("PASS: all %0d vectors x %0d columns match", NVEC, N);
        else
            $display("FAIL: %0d errors, last column drained %0d/%0d vectors",
                     errors, vec_idx[N-1], NVEC);
        $finish;
    end

endmodule
