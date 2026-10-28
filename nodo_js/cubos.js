'use strict';

// JavaScript solo tiene float64. Para dar exactamente lo mismo que Java (float32),
// cada producto y cada suma se redondea con Math.fround, en el mismo orden que
// CuboConvolucion.java y CuboMLP.java. Escribir en un Float32Array ya redondea.
const fr = Math.fround;

function convForward(entrada, C, H, W, p, wOff, bOff, fIni, fFin, z) {
    const hw = H * W;
    for (let f = fIni; f < fFin; f++) {
        const zo = f * hw;
        z.fill(p[bOff + f], zo, zo + hw);
        for (let c = 0; c < C; c++) {
            const io = c * hw;
            const wo = wOff + (f * C + c) * 9;
            for (let ky = 0; ky < 3; ky++) {
                const dy = ky - 1;
                const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
                for (let kx = 0; kx < 3; kx++) {
                    const dx = kx - 1;
                    const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
                    const wv = p[wo + ky * 3 + kx];
                    for (let y = y0; y < y1; y++) {
                        const zr = zo + y * W;
                        const ir = io + (y + dy) * W + dx;
                        for (let x = x0; x < x1; x++) {
                            z[zr + x] += fr(wv * entrada[ir + x]);
                        }
                    }
                }
            }
        }
    }
}

function reluPool(z, fIni, fFin, H, W, salida, idx) {
    const H2 = H >> 1, W2 = W >> 1, hw = H * W;
    for (let f = fIni; f < fFin; f++) {
        for (let py = 0; py < H2; py++) {
            for (let px = 0; px < W2; px++) {
                const base = f * hw + (2 * py) * W + 2 * px;
                let m = base;
                let v = z[base];
                if (z[base + 1] > v) { v = z[base + 1]; m = base + 1; }
                if (z[base + W] > v) { v = z[base + W]; m = base + W; }
                if (z[base + W + 1] > v) { v = z[base + W + 1]; m = base + W + 1; }
                const o = f * H2 * W2 + py * W2 + px;
                salida[o] = v > 0 ? v : 0;
                if (idx !== null) {
                    idx[o] = m;
                }
            }
        }
    }
}

function reluPoolBackward(dSalida, idx, z, F, H, W, dz) {
    dz.fill(0, 0, F * H * W);
    const n = F * (H >> 1) * (W >> 1);
    for (let o = 0; o < n; o++) {
        const i = idx[o];
        if (z[i] > 0) {
            dz[i] += dSalida[o];
        }
    }
}

function convBackward(entrada, C, H, W, dz, fIni, fFin, p, wOff, bOff, g, dEntrada) {
    const hw = H * W;
    for (let f = fIni; f < fFin; f++) {
        const zo = f * hw;
        let sb = 0;
        for (let i = 0; i < hw; i++) {
            sb = fr(sb + dz[zo + i]);
        }
        g[bOff + f] += sb;
        for (let c = 0; c < C; c++) {
            const io = c * hw;
            const wo = wOff + (f * C + c) * 9;
            for (let ky = 0; ky < 3; ky++) {
                const dy = ky - 1;
                const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
                for (let kx = 0; kx < 3; kx++) {
                    const dx = kx - 1;
                    const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
                    const wv = p[wo + ky * 3 + kx];
                    let acc = 0;
                    for (let y = y0; y < y1; y++) {
                        const zr = zo + y * W;
                        const ir = io + (y + dy) * W + dx;
                        if (dEntrada !== null) {
                            for (let x = x0; x < x1; x++) {
                                const d = dz[zr + x];
                                acc = fr(acc + fr(d * entrada[ir + x]));
                                dEntrada[ir + x] += fr(wv * d);
                            }
                        } else {
                            for (let x = x0; x < x1; x++) {
                                acc = fr(acc + fr(dz[zr + x] * entrada[ir + x]));
                            }
                        }
                    }
                    g[wo + ky * 3 + kx] += acc;
                }
            }
        }
    }
}

function densaForward(x, nIn, p, wOff, bOff, jIni, jFin, salida, relu) {
    for (let j = jIni; j < jFin; j++) {
        const wo = wOff + j * nIn;
        let s = 0;
        for (let i = 0; i < nIn; i++) {
            s = fr(s + fr(p[wo + i] * x[i]));
        }
        s = fr(s + p[bOff + j]);
        salida[j] = (relu && s < 0) ? 0 : s;
    }
}

function densaBackward(x, nIn, dSalida, nOut, p, wOff, bOff, g, dEntrada) {
    for (let j = 0; j < nOut; j++) {
        const d = dSalida[j];
        if (d === 0) {
            continue;
        }
        g[bOff + j] += d;
        const wo = wOff + j * nIn;
        for (let i = 0; i < nIn; i++) {
            g[wo + i] += fr(d * x[i]);
        }
        if (dEntrada !== null) {
            for (let i = 0; i < nIn; i++) {
                dEntrada[i] += fr(p[wo + i] * d);
            }
        }
    }
}

// Igual que CuboMLP.softmax: maximo en float32, exponenciales y suma en double.
function softmax(z, n, prob) {
    let max = z[0];
    for (let i = 1; i < n; i++) {
        if (z[i] > max) {
            max = z[i];
        }
    }
    let suma = 0;
    for (let i = 0; i < n; i++) {
        const e = Math.exp(fr(z[i] - max));
        prob[i] = e;
        suma += e;
    }
    for (let i = 0; i < n; i++) {
        prob[i] = prob[i] / suma;
    }
}

module.exports = { convForward, reluPool, reluPoolBackward, convBackward, densaForward, densaBackward, softmax };
