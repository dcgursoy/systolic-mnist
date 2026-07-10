// Cycle-accurate JavaScript twin of the accelerator RTL (batch = 1).
//
// This is a register-for-register model of rtl/accel_top.sv and its
// submodules: same FSM, same synchronous memory reads, same skew pipelines,
// same swap-token double buffering, same 2-stage requantize pipeline. Each
// simulated cycle produces a frame identical in content to what the Icarus
// Verilog testbench dumps at the negedge — viz/validate_sim.py diffs the
// two, frame by frame, against sim/traces/img0.json.
//
// Update discipline mirrors nonblocking assignment: phase A samples all
// combinational values from current register state (this is where frames
// are emitted, like the TB's negedge sampling); phase B computes every
// next-register value from current state, then commits atomically.

window.AccelSim = (function () {
  "use strict";

  const S_IDLE = 0, S_SETUP = 1, S_PRELOAD = 2, S_RUN = 3, S_TAIL = 4,
        S_REQ = 5, S_DONE = 6;

  function simulate(img, M, opts) {
    opts = opts || {};
    const record = opts.record !== false;
    const N = M.n, BATCH = 1;
    const DEPTH = 196;

    // ---- registers ----
    let state = S_IDLE, layer = 0, outBlk = 0, inTile = 0;
    let wPtr = 0, bPtr = 0, slot = 0, issueB = 0;
    let blockDrainCnt = 0, targetDrains = 0, reqCnt = 0;
    let wlRun = false, wlCnt = 0, loadedNext = false;
    let vecValidD = false, firstD = false, tokenD = false;
    let reqWenD = false, reqWenD2 = false, reqBD = 0, reqBD2 = 0;
    let cycleCount = 0, done = false, start = true;

    let wdataR = [0, 0, 0, 0].slice(0, N);
    let biasWordR = new Array(N).fill(0);

    const actMem = [0, 1].map(() =>
      Array.from({length: N}, () => new Array(DEPTH).fill(0)));
    let actRdata = [0, 1].map(() => new Array(N).fill(0));

    // skew pipelines (row r has r stages; row 0 is passthrough)
    const pipeA = Array.from({length: N}, (_, r) => new Array(r).fill(0));
    const pipeT = Array.from({length: N}, (_, r) => new Array(r).fill(0));

    const mk2d = v => Array.from({length: N},
                                 () => new Array(N).fill(v));
    let wShadow = mk2d(0), wActive = mk2d(0), aOut = mk2d(0),
        psumOut = mk2d(0), swapOut = mk2d(false);

    const shV = Array.from({length: N}, (_, c) => new Array(N + c).fill(0));
    const shF = Array.from({length: N}, (_, c) => new Array(N + c).fill(0));

    const accMem = Array.from({length: N}, () => new Array(16).fill(0));
    const accWcnt = new Array(N).fill(0);
    let accRdata = new Array(N).fill(0);
    let accRdataD = new Array(N).fill(0);
    let prodQ = new Array(N).fill(0);

    // image -> parity-0 activation banks (element idx: bank idx%N, addr idx/N)
    for (let px = 0; px < 784; px++)
      actMem[0][px % N][(px / N) | 0] = img[px];

    const frames = [];
    const logits = new Array(10).fill(null);
    const shift2 = s => Math.pow(2, s);

    let guard = 200000;
    while (guard-- > 0) {
      // ================= phase A: combinational + sampling ==============
      const nTiles = M.inTiles[layer], nBlocks = M.outBlocks[layer];
      const lastLayer = (layer === M.layerDims.length - 1);
      const lastTile = (inTile === nTiles - 1);
      const rdParity = layer & 1, wrParity = 1 - rdParity;

      const aAligned = [], aSkewed = [], swapSkewed = [];
      for (let r = 0; r < N; r++) {
        aAligned[r] = actRdata[rdParity][r];
        aSkewed[r] = r === 0 ? aAligned[0] : pipeA[r][r - 1];
        swapSkewed[r] = r === 0 ? (tokenD ? 1 : 0) : pipeT[r][r - 1];
      }
      const aH = mk2d(0), swapH = mk2d(0);
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++) {
          aH[r][c] = c === 0 ? aSkewed[r] : aOut[r][c - 1];
          swapH[r][c] = c === 0 ? swapSkewed[r] : (swapOut[r][c - 1] ? 1 : 0);
        }
      const drainValid = [], drainFirst = [];
      for (let c = 0; c < N; c++) {
        drainValid[c] = shV[c][N + c - 1];
        drainFirst[c] = shF[c][N + c - 1];
      }
      const psumBottom = [];
      for (let c = 0; c < N; c++) psumBottom[c] = psumOut[N - 1][c];

      const wLoadEn = wlRun && wlCnt !== 0;
      const wRowSel = wlCnt - 1;
      const blockStart = (state === S_PRELOAD) && (wlCnt === 0);

      const reqY = [];
      for (let c = 0; c < N; c++) {
        const sh = Math.floor(prodQ[c] / shift2(M.shift[layer]));
        reqY[c] = sh < 0 ? 0 : sh > 127 ? 127 : sh;
      }
      const logitsValid = reqWenD2 && lastLayer;
      const actWrite = reqWenD2 && !lastLayer;

      // ---- sampling (mirrors the TB's negedge) ----
      if (state !== S_IDLE) {
        if (logitsValid)
          for (let c = 0; c < N; c++) {
            const idx = outBlk * N + c;
            if (idx < 10) logits[idx] = accRdataD[c];
          }
        if (record) {
          const pe = [];
          for (let r = 0; r < N; r++) {
            pe[r] = [];
            for (let c = 0; c < N; c++)
              pe[r][c] = {a: aH[r][c], w: wActive[r][c], p: psumOut[r][c]};
          }
          let dv = 0;
          for (let c = 0; c < N; c++) dv |= drainValid[c] << c;
          const f = {c: cycleCount, st: state, l: layer, t: inTile,
                     j: outBlk, pe: pe, dv: dv,
                     acc: accMem.map(mem => mem[0])};
          if (reqWenD2)
            f.req = {b: reqBD2, y: reqY.slice(), acc: accRdataD.slice()};
          frames.push(f);
        }
        if (state === S_DONE) break;
      }

      // ================= phase B: next-state computation ================
      const nx = {
        state, layer, outBlk, inTile, wPtr, bPtr, slot, issueB,
        blockDrainCnt, targetDrains, reqCnt, wlRun, wlCnt, loadedNext,
        cycleCount, done,
      };

      // stream-issue pipeline flags
      const nVecValidD = (state === S_RUN) && (slot >= 1) && (issueB < BATCH);
      const nFirstD = nVecValidD && (inTile === 0);
      const nTokenD = (state === S_RUN) && (slot === 0);

      // requantize write pipeline
      const nReqWenD = (state === S_REQ) && (reqCnt < BATCH);
      const nReqBD = reqCnt & 15;
      const nReqWenD2 = reqWenD, nReqBD2 = reqBD;
      const nAccRdataD = accRdata.slice();
      const nProdQ = [];
      for (let c = 0; c < N; c++)
        nProdQ[c] = accRdata[c] * M.M[layer]
                    + shift2(M.shift[layer] - 1);

      // synchronous memory reads (old memory contents)
      const nWdataR = M.wmem[Math.min(wPtr, M.wmem.length - 1)].slice();
      const nBiasWordR = M.bmem[Math.min(bPtr, M.bmem.length - 1)].slice();
      // act-bank read address includes the batch lane (rb = issue_b);
      // lanes >= 1 are never written at batch 1, so they read as zero
      const nActRdata = [0, 1].map(p =>
        Array.from({length: N}, (_, r) =>
          issueB === 0 ? actMem[p][r][Math.min(inTile, DEPTH - 1)] : 0));
      const nAccRdata = Array.from({length: N},
                                   (_, c) => accMem[c][reqCnt & 15]);

      // PE array
      const nWShadow = mk2d(0), nWActive = mk2d(0), nAOut = mk2d(0),
            nPsumOut = mk2d(0), nSwapOut = mk2d(false);
      for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++) {
          nWShadow[r][c] = (wLoadEn && wRowSel === r)
                           ? wdataR[c] : wShadow[r][c];
          nWActive[r][c] = swapH[r][c] ? wShadow[r][c] : wActive[r][c];
          nSwapOut[r][c] = !!swapH[r][c];
          nAOut[r][c] = aH[r][c];
          nPsumOut[r][c] = (r === 0 ? 0 : psumOut[r - 1][c])
                           + aH[r][c] * wActive[r][c];
        }

      // skew pipelines
      const nPipeA = pipeA.map((p, r) =>
        r === 0 ? [] : [aAligned[r]].concat(p.slice(0, r - 1)));
      const nPipeT = pipeT.map((p, r) =>
        r === 0 ? [] : [tokenD ? 1 : 0].concat(p.slice(0, r - 1)));

      // valid chains
      const nShV = shV.map((s, c) =>
        [vecValidD ? 1 : 0].concat(s.slice(0, N + c - 1)));
      const nShF = shF.map((s, c) =>
        [firstD ? 1 : 0].concat(s.slice(0, N + c - 1)));

      // accumulator banks (writes commit after the read sampling above)
      const nAccWcnt = accWcnt.slice();
      for (let c = 0; c < N; c++) {
        if (blockStart) {
          nAccWcnt[c] = 0;
        } else if (drainValid[c]) {
          const w = accWcnt[c] & 15;
          accMem[c][w] = (drainFirst[c] ? biasWordR[c] : accMem[c][w])
                         + psumBottom[c];
          nAccWcnt[c] = (accWcnt[c] === BATCH - 1) ? 0 : accWcnt[c] + 1;
        }
      }

      // activation-bank writes (requantize results; image preloaded)
      if (actWrite)
        for (let c = 0; c < N; c++)
          actMem[wrParity][c][outBlk] = reqY[c];

      // shadow-load engine
      if (wlRun) {
        if (wlCnt < N) nx.wPtr = wPtr + 1;
        nx.wlCnt = wlCnt + 1;
        if (wlCnt === N) { nx.wlRun = false; nx.loadedNext = true; }
      }
      if (drainValid[N - 1]) nx.blockDrainCnt = blockDrainCnt + 1;
      if (state !== S_IDLE && state !== S_DONE)
        nx.cycleCount = cycleCount + 1;

      // FSM (case-branch assignments override the engine defaults above)
      switch (state) {
        case S_IDLE:
          if (start) {
            start = false;
            nx.layer = 0; nx.outBlk = 0; nx.wPtr = 0; nx.bPtr = 0;
            nx.done = false; nx.cycleCount = 0; nx.state = S_SETUP;
          }
          break;
        case S_SETUP:
          nx.wlRun = true; nx.wlCnt = 0; nx.loadedNext = false;
          nx.blockDrainCnt = 0;
          nx.targetDrains = nTiles * BATCH;
          nx.state = S_PRELOAD;
          break;
        case S_PRELOAD:
          if (wlCnt === N) {
            nx.inTile = 0; nx.slot = 0; nx.issueB = 0;
            nx.loadedNext = false; nx.state = S_RUN;
          }
          break;
        case S_RUN:
          nx.slot = slot + 1;
          if (slot >= 1 && issueB < BATCH) nx.issueB = issueB + 1;
          if (slot === N - 1 && !lastTile && !wlRun && !loadedNext) {
            nx.wlRun = true; nx.wlCnt = 0;
          }
          if (issueB === BATCH) {
            if (lastTile) nx.state = S_TAIL;
            else if (loadedNext) {
              nx.inTile = inTile + 1; nx.slot = 0; nx.issueB = 0;
              nx.loadedNext = false;
            }
          }
          break;
        case S_TAIL:
          if (blockDrainCnt === targetDrains) {
            nx.reqCnt = 0; nx.state = S_REQ;
          }
          break;
        case S_REQ:
          nx.reqCnt = reqCnt + 1;
          if (reqCnt === BATCH + 1) {
            nx.bPtr = bPtr + 1;
            if (outBlk === nBlocks - 1) {
              if (lastLayer) { nx.done = true; nx.state = S_DONE; }
              else { nx.layer = layer + 1; nx.outBlk = 0; nx.state = S_SETUP; }
            } else { nx.outBlk = outBlk + 1; nx.state = S_SETUP; }
          }
          break;
        default:
          nx.state = S_IDLE;
      }

      // ================= commit =================
      state = nx.state; layer = nx.layer; outBlk = nx.outBlk;
      inTile = nx.inTile; wPtr = nx.wPtr; bPtr = nx.bPtr; slot = nx.slot;
      issueB = nx.issueB; blockDrainCnt = nx.blockDrainCnt;
      targetDrains = nx.targetDrains; reqCnt = nx.reqCnt;
      wlRun = nx.wlRun; wlCnt = nx.wlCnt; loadedNext = nx.loadedNext;
      cycleCount = nx.cycleCount; done = nx.done;
      vecValidD = nVecValidD; firstD = nFirstD; tokenD = nTokenD;
      reqWenD = nReqWenD; reqBD = nReqBD;
      reqWenD2 = nReqWenD2; reqBD2 = nReqBD2;
      accRdataD = nAccRdataD; prodQ = nProdQ;
      wdataR = nWdataR; biasWordR = nBiasWordR; actRdata = nActRdata;
      accRdata = nAccRdata;
      wShadow = nWShadow; wActive = nWActive; aOut = nAOut;
      psumOut = nPsumOut; swapOut = nSwapOut;
      for (let r = 0; r < N; r++) {
        pipeA[r] = nPipeA[r]; pipeT[r] = nPipeT[r];
        shV[r] = nShV[r]; shF[r] = nShF[r];
      }
      for (let c = 0; c < N; c++) accWcnt[c] = nAccWcnt[c];
    }

    return {frames, logits, cycles: cycleCount};
  }

  // float reference from dequantized INT8 weights (for user-drawn digits)
  function floatForward(img, M) {
    let x = Array.from(img, v => v / 127.0);
    const dims = M.layerDims;
    for (let l = 0; l < dims.length; l++) {
      const [inF, outF] = dims[l];
      const y = new Array(outF).fill(0);
      // reconstruct W and b from the packed memories
      const tilesPerBlock = M.inTiles[l];
      let base = 0;
      for (let ll = 0; ll < l; ll++)
        base += M.inTiles[ll] * M.outBlocks[ll] * M.n;
      let bBase = 0;
      for (let ll = 0; ll < l; ll++) bBase += M.outBlocks[ll];
      for (let j = 0; j < M.outBlocks[l]; j++)
        for (let i = 0; i < tilesPerBlock; i++)
          for (let r = 0; r < M.n; r++) {
            const word = M.wmem[base + (j * tilesPerBlock + i) * M.n + r];
            for (let c = 0; c < M.n; c++) {
              const out = j * M.n + c;
              if (out < outF)
                y[out] += word[c] * M.wScale[l] * x[i * M.n + r];
            }
          }
      for (let j = 0; j < M.outBlocks[l]; j++)
        for (let c = 0; c < M.n; c++) {
          const out = j * M.n + c;
          if (out < outF)
            y[out] += M.bmem[bBase + j][c] * M.actScale[l] * M.wScale[l];
        }
      x = l < dims.length - 1 ? y.map(v => Math.max(0, v)) : y;
    }
    return x;
  }

  return {simulate, floatForward};
})();
