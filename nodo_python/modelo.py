import math

import cubos

class Arquitectura:
    def __init__(self, c, h, w, f1, f2, oculta, k):
        self.c, self.h, self.w, self.f1, self.f2, self.oculta, self.k = c, h, w, f1, f2, oculta, k
        self.h2, self.w2, self.h4, self.w4 = h // 2, w // 2, h // 4, w // 4
        self.plano = f2 * self.h4 * self.w4
        self.oW1 = 0
        self.oB1 = self.oW1 + f1 * c * 9
        self.oW2 = self.oB1 + f1
        self.oB2 = self.oW2 + f2 * f1 * 9
        self.oW3 = self.oB2 + f2
        self.oB3 = self.oW3 + oculta * self.plano
        self.oW4 = self.oB3 + oculta
        self.oB4 = self.oW4 + k * oculta
        self.total = self.oB4 + k

    def tupla(self):
        return (self.c, self.h, self.w, self.f1, self.f2, self.oculta, self.k)

    def tam_imagen(self):
        return self.c * self.h * self.w

def normalizar(img):
    return [b / 255.0 - 0.5 for b in img]

def _forward(p, a, x):
    cols1 = cubos.im2col(x, a.c, a.h, a.w)
    z1 = cubos.conv_forward(cols1, p, a.oW1, a.oB1, a.c, 0, a.f1)
    a1, i1 = cubos.relu_pool(z1, a.h, a.w)
    cols2 = cubos.im2col(a1, a.f1, a.h2, a.w2)
    z2 = cubos.conv_forward(cols2, p, a.oW2, a.oB2, a.f1, 0, a.f2)
    a2, i2 = cubos.relu_pool(z2, a.h2, a.w2)
    z3 = cubos.densa_forward(a2, p, a.oW3, a.oB3, a.plano, 0, a.oculta, False)
    a3 = [v if v > 0.0 else 0.0 for v in z3]
    o = cubos.densa_forward(a3, p, a.oW4, a.oB4, a.oculta, 0, a.k, False)
    return {'cols1': cols1, 'z1': z1, 'i1': i1, 'a1': a1, 'cols2': cols2, 'z2': z2, 'i2': i2,
            'a2': a2, 'z3': z3, 'a3': a3, 'prob': cubos.softmax(o)}

def predecir(p, a, img):
    return _forward(p, a, normalizar(img))['prob']

def entrenar_muestra(p, a, img, etiqueta, g):
    c = _forward(p, a, normalizar(img))
    prob = c['prob']
    acierto = max(range(a.k), key=lambda i: prob[i]) == etiqueta
    perdida = -math.log(prob[etiqueta] + 1e-9)
    d_o = [prob[i] - (1.0 if i == etiqueta else 0.0) for i in range(a.k)]
    d_a3 = cubos.densa_backward(c['a3'], d_o, p, a.oW4, a.oB4, a.oculta, g, True)
    d_z3 = [d if z > 0.0 else 0.0 for d, z in zip(d_a3, c['z3'])]
    d_a2 = cubos.densa_backward(c['a2'], d_z3, p, a.oW3, a.oB3, a.plano, g, True)
    d_z2 = cubos.relu_pool_backward(d_a2, c['i2'], c['z2'], a.h2, a.w2)
    d_a1 = cubos.conv_backward(c['cols2'], d_z2, p, a.oW2, a.oB2, a.f1, g, True, a.h2, a.w2)
    d_z1 = cubos.relu_pool_backward(d_a1, c['i1'], c['z1'], a.h, a.w)
    cubos.conv_backward(c['cols1'], d_z1, p, a.oW1, a.oB1, a.c, g, False, a.h, a.w)
    return perdida, acierto

def entrenar_lote(p, a, imgs, etiquetas):
    tam = a.tam_imagen()
    g = [0.0] * a.total
    perdida = 0.0
    aciertos = 0
    for i, e in enumerate(etiquetas):
        l, ok = entrenar_muestra(p, a, imgs[i * tam:(i + 1) * tam], e, g)
        perdida += l
        aciertos += 1 if ok else 0
    return g, perdida, aciertos

def inferir_lote(p, a, imgs, n):
    tam = a.tam_imagen()
    probs = []
    for i in range(n):
        probs.extend(predecir(p, a, imgs[i * tam:(i + 1) * tam]))
    return probs

def conv_micro(p, a, capa, f_ini, f_fin, entrada):
    if capa == 1:
        C, H, W, w_off, b_off = a.c, a.h, a.w, a.oW1, a.oB1
    else:
        C, H, W, w_off, b_off = a.f1, a.h2, a.w2, a.oW2, a.oB2
    cols = cubos.im2col(entrada, C, H, W)
    z = cubos.conv_forward(cols, p, w_off, b_off, C, f_ini, f_fin)
    out, _ = cubos.relu_pool(z, H, W)
    return out

def mlp_micro(p, a, capa, j_ini, j_fin, entrada):
    if capa == 1:
        return cubos.densa_forward(entrada, p, a.oW3, a.oB3, a.plano, j_ini, j_fin, True)
    return cubos.densa_forward(entrada, p, a.oW4, a.oB4, a.oculta, j_ini, j_fin, False)
