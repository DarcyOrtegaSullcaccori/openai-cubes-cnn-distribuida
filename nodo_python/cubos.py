import math
from itertools import repeat
from operator import add, mul

def im2col(x, C, H, W):
    ancho = W + 2
    cols = []
    for c in range(C):
        canal = x[c * H * W:(c + 1) * H * W]
        pad = [0.0] * ((H + 2) * ancho)
        for y in range(H):
            ini = (y + 1) * ancho + 1
            pad[ini:ini + W] = canal[y * W:(y + 1) * W]
        for ky in range(3):
            for kx in range(3):
                col = []
                for y in range(H):
                    ini = (y + ky) * ancho + kx
                    col.extend(pad[ini:ini + W])
                cols.append(col)
    return cols

def conv_forward(cols, p, w_off, b_off, C, f_ini, f_fin):
    k = C * 9
    parches = list(zip(*cols))
    salida = []
    for f in range(f_ini, f_fin):
        wf = p[w_off + f * k: w_off + (f + 1) * k]
        b = p[b_off + f]
        salida.append([b + sum(map(mul, wf, pt)) for pt in parches])
    return salida

def relu_pool(z_filas, H, W):
    H2, W2 = H // 2, W // 2
    out = []
    idx = []
    for z in z_filas:
        for py in range(H2):
            r0 = 2 * py * W
            r1 = r0 + W
            for px in range(W2):
                a = r0 + 2 * px
                m = a
                v = z[a]
                if z[a + 1] > v:
                    v = z[a + 1]
                    m = a + 1
                b = r1 + 2 * px
                if z[b] > v:
                    v = z[b]
                    m = b
                if z[b + 1] > v:
                    v = z[b + 1]
                    m = b + 1
                out.append(v if v > 0.0 else 0.0)
                idx.append(m)
    return out, idx

def relu_pool_backward(d_out, idx, z_filas, H, W):
    n = (H // 2) * (W // 2)
    dz = []
    for f, z in enumerate(z_filas):
        d = [0.0] * (H * W)
        base = f * n
        for o in range(n):
            g = d_out[base + o]
            if g != 0.0:
                i = idx[base + o]
                if z[i] > 0.0:
                    d[i] += g
        dz.append(d)
    return dz

def conv_backward(cols, dz_filas, p, w_off, b_off, C, g, calcular_din, H, W):
    k = C * 9
    for f, dz in enumerate(dz_filas):
        if not any(dz):
            continue
        g[b_off + f] += sum(dz)
        base = w_off + f * k
        for j in range(k):
            g[base + j] += sum(map(mul, dz, cols[j]))
    if not calcular_din:
        return None
    F = len(dz_filas)
    ancho = W + 2
    dz_t = list(zip(*dz_filas))
    dpad = [[0.0] * ((H + 2) * ancho) for _ in range(C)]
    for j in range(k):
        wcol = [p[w_off + f * k + j] for f in range(F)]
        dcol = [sum(map(mul, wcol, t)) for t in dz_t]
        c, r = divmod(j, 9)
        ky, kx = divmod(r, 3)
        pad = dpad[c]
        for y in range(H):
            ini = (y + ky) * ancho + kx
            pad[ini:ini + W] = list(map(add, pad[ini:ini + W], dcol[y * W:(y + 1) * W]))
    din = []
    for c in range(C):
        for y in range(H):
            ini = (y + 1) * ancho + 1
            din.extend(dpad[c][ini:ini + W])
    return din

def densa_forward(x, p, w_off, b_off, n_in, j_ini, j_fin, relu):
    out = []
    for j in range(j_ini, j_fin):
        wo = w_off + j * n_in
        s = p[b_off + j] + sum(map(mul, p[wo:wo + n_in], x))
        out.append(0.0 if (relu and s < 0.0) else s)
    return out

def densa_backward(x, d_out, p, w_off, b_off, n_in, g, calcular_din):
    din = [0.0] * n_in if calcular_din else None
    for j, d in enumerate(d_out):
        if d == 0.0:
            continue
        g[b_off + j] += d
        wo = w_off + j * n_in
        g[wo:wo + n_in] = list(map(add, g[wo:wo + n_in], map(mul, x, repeat(d))))
        if calcular_din:
            din = list(map(add, din, map(mul, p[wo:wo + n_in], repeat(d))))
    return din

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]
