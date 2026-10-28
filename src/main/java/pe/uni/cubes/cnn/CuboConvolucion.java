package pe.uni.cubes.cnn;

public final class CuboConvolucion {

    private CuboConvolucion() {
    }

    public static void forward(float[] in, int C, int H, int W, float[] p, int wOff, int bOff,
                               int fIni, int fFin, float[] z) {
        int hw = H * W;
        for (int f = fIni; f < fFin; f++) {
            int zo = f * hw;
            float b = p[bOff + f];
            for (int i = 0; i < hw; i++) {
                z[zo + i] = b;
            }
            for (int c = 0; c < C; c++) {
                int io = c * hw;
                int wo = wOff + (f * C + c) * 9;
                for (int ky = 0; ky < 3; ky++) {
                    int dy = ky - 1;
                    int y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
                    for (int kx = 0; kx < 3; kx++) {
                        int dx = kx - 1;
                        int x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
                        float wv = p[wo + ky * 3 + kx];
                        for (int y = y0; y < y1; y++) {
                            int zr = zo + y * W;
                            int ir = io + (y + dy) * W + dx;
                            for (int x = x0; x < x1; x++) {
                                z[zr + x] += wv * in[ir + x];
                            }
                        }
                    }
                }
            }
        }
    }

    public static void reluPool(float[] z, int fIni, int fFin, int H, int W, float[] out, int[] idx) {
        int H2 = H / 2, W2 = W / 2, hw = H * W;
        for (int f = fIni; f < fFin; f++) {
            for (int py = 0; py < H2; py++) {
                for (int px = 0; px < W2; px++) {
                    int base = f * hw + (2 * py) * W + 2 * px;
                    int mejor = base;
                    float v = z[base];
                    if (z[base + 1] > v) { v = z[base + 1]; mejor = base + 1; }
                    if (z[base + W] > v) { v = z[base + W]; mejor = base + W; }
                    if (z[base + W + 1] > v) { v = z[base + W + 1]; mejor = base + W + 1; }
                    int o = f * H2 * W2 + py * W2 + px;
                    out[o] = v > 0f ? v : 0f;
                    if (idx != null) {
                        idx[o] = mejor;
                    }
                }
            }
        }
    }

    public static void reluPoolBackward(float[] dOut, int[] idx, float[] z, int F, int H, int W, float[] dz) {
        java.util.Arrays.fill(dz, 0, F * H * W, 0f);
        int n = F * (H / 2) * (W / 2);
        for (int o = 0; o < n; o++) {
            int i = idx[o];
            if (z[i] > 0f) {
                dz[i] += dOut[o];
            }
        }
    }

    public static void backward(float[] in, int C, int H, int W, float[] dz, int fIni, int fFin,
                                float[] p, int wOff, int bOff, float[] g, float[] dIn) {
        int hw = H * W;
        for (int f = fIni; f < fFin; f++) {
            int zo = f * hw;
            float sb = 0f;
            for (int i = 0; i < hw; i++) {
                sb += dz[zo + i];
            }
            g[bOff + f] += sb;
            for (int c = 0; c < C; c++) {
                int io = c * hw;
                int wo = wOff + (f * C + c) * 9;
                for (int ky = 0; ky < 3; ky++) {
                    int dy = ky - 1;
                    int y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
                    for (int kx = 0; kx < 3; kx++) {
                        int dx = kx - 1;
                        int x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
                        float wv = p[wo + ky * 3 + kx];
                        float acc = 0f;
                        for (int y = y0; y < y1; y++) {
                            int zr = zo + y * W;
                            int ir = io + (y + dy) * W + dx;
                            if (dIn != null) {
                                for (int x = x0; x < x1; x++) {
                                    float d = dz[zr + x];
                                    acc += d * in[ir + x];
                                    dIn[ir + x] += wv * d;
                                }
                            } else {
                                for (int x = x0; x < x1; x++) {
                                    acc += dz[zr + x] * in[ir + x];
                                }
                            }
                        }
                        g[wo + ky * 3 + kx] += acc;
                    }
                }
            }
        }
    }
}
