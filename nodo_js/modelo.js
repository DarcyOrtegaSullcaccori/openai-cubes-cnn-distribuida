'use strict';

const cubos = require('./cubos');

const fr = Math.fround;

class Arquitectura {
    constructor(c, h, w, f1, f2, oculta, k) {
        Object.assign(this, { c, h, w, f1, f2, oculta, k });
        this.h2 = h >> 1;
        this.w2 = w >> 1;
        this.h4 = h >> 2;
        this.w4 = w >> 2;
        this.plano = f2 * this.h4 * this.w4;
        this.oW1 = 0;
        this.oB1 = this.oW1 + f1 * c * 9;
        this.oW2 = this.oB1 + f1;
        this.oB2 = this.oW2 + f2 * f1 * 9;
        this.oW3 = this.oB2 + f2;
        this.oB3 = this.oW3 + oculta * this.plano;
        this.oW4 = this.oB3 + oculta;
        this.oB4 = this.oW4 + k * oculta;
        this.total = this.oB4 + k;
    }

    tupla() {
        return [this.c, this.h, this.w, this.f1, this.f2, this.oculta, this.k];
    }

    tamImagen() {
        return this.c * this.h * this.w;
    }
}

function nuevoEspacio(a) {
    const e = {};
    e.x = new Float32Array(a.c * a.h * a.w);
    e.z1 = new Float32Array(a.f1 * a.h * a.w);
    e.a1 = new Float32Array(a.f1 * a.h2 * a.w2);
    e.i1 = new Int32Array(e.a1.length);
    e.z2 = new Float32Array(a.f2 * a.h2 * a.w2);
    e.a2 = new Float32Array(a.plano);
    e.i2 = new Int32Array(e.a2.length);
    e.z3 = new Float32Array(a.oculta);
    e.a3 = new Float32Array(a.oculta);
    e.o = new Float32Array(a.k);
    e.prob = new Float32Array(a.k);
    e.dO = new Float32Array(a.k);
    e.dA3 = new Float32Array(a.oculta);
    e.dZ3 = new Float32Array(a.oculta);
    e.dA2 = new Float32Array(a.plano);
    e.dZ2 = new Float32Array(e.z2.length);
    e.dA1 = new Float32Array(e.a1.length);
    e.dZ1 = new Float32Array(e.z1.length);
    return e;
}

function normalizar(imgs, off, x) {
    for (let i = 0; i < x.length; i++) {
        x[i] = fr(imgs[off + i] / 255) - 0.5;
    }
}

function forward(a, p, e) {
    cubos.convForward(e.x, a.c, a.h, a.w, p, a.oW1, a.oB1, 0, a.f1, e.z1);
    cubos.reluPool(e.z1, 0, a.f1, a.h, a.w, e.a1, e.i1);
    cubos.convForward(e.a1, a.f1, a.h2, a.w2, p, a.oW2, a.oB2, 0, a.f2, e.z2);
    cubos.reluPool(e.z2, 0, a.f2, a.h2, a.w2, e.a2, e.i2);
    cubos.densaForward(e.a2, a.plano, p, a.oW3, a.oB3, 0, a.oculta, e.z3, false);
    for (let j = 0; j < a.oculta; j++) {
        e.a3[j] = e.z3[j] > 0 ? e.z3[j] : 0;
    }
    cubos.densaForward(e.a3, a.oculta, p, a.oW4, a.oB4, 0, a.k, e.o, false);
    cubos.softmax(e.o, a.k, e.prob);
}

function predecir(a, p, imgs, off, probs, pOff, e) {
    normalizar(imgs, off, e.x);
    forward(a, p, e);
    probs.set(e.prob, pOff);
}

function entrenarMuestra(a, p, imgs, off, etiqueta, g, e) {
    normalizar(imgs, off, e.x);
    forward(a, p, e);
    let mejor = 0;
    for (let i = 1; i < a.k; i++) {
        if (e.prob[i] > e.prob[mejor]) {
            mejor = i;
        }
    }
    const perdida = fr(-Math.log(e.prob[etiqueta] + 1e-9));
    for (let i = 0; i < a.k; i++) {
        e.dO[i] = e.prob[i] - (i === etiqueta ? 1 : 0);
    }
    e.dA3.fill(0);
    cubos.densaBackward(e.a3, a.oculta, e.dO, a.k, p, a.oW4, a.oB4, g, e.dA3);
    for (let j = 0; j < a.oculta; j++) {
        e.dZ3[j] = e.z3[j] > 0 ? e.dA3[j] : 0;
    }
    e.dA2.fill(0);
    cubos.densaBackward(e.a2, a.plano, e.dZ3, a.oculta, p, a.oW3, a.oB3, g, e.dA2);
    cubos.reluPoolBackward(e.dA2, e.i2, e.z2, a.f2, a.h2, a.w2, e.dZ2);
    e.dA1.fill(0);
    cubos.convBackward(e.a1, a.f1, a.h2, a.w2, e.dZ2, 0, a.f2, p, a.oW2, a.oB2, g, e.dA1);
    cubos.reluPoolBackward(e.dA1, e.i1, e.z1, a.f1, a.h, a.w, e.dZ1);
    cubos.convBackward(e.x, a.c, a.h, a.w, e.dZ1, 0, a.f1, p, a.oW1, a.oB1, g, null);
    return { perdida, acierto: mejor === etiqueta };
}

function entrenarLote(a, p, imgs, etiquetas, e) {
    const tam = a.tamImagen();
    const g = new Float32Array(a.total);
    let perdida = 0, aciertos = 0;
    for (let i = 0; i < etiquetas.length; i++) {
        const r = entrenarMuestra(a, p, imgs, i * tam, etiquetas[i], g, e);
        perdida = fr(perdida + r.perdida);
        if (r.acierto) {
            aciertos++;
        }
    }
    return { g, perdida, aciertos };
}

function inferirLote(a, p, imgs, n, e) {
    const tam = a.tamImagen();
    const probs = new Float32Array(n * a.k);
    for (let i = 0; i < n; i++) {
        predecir(a, p, imgs, i * tam, probs, i * a.k, e);
    }
    return probs;
}

function convMicro(a, p, capa, fIni, fFin, entrada) {
    const C = capa === 1 ? a.c : a.f1;
    const H = capa === 1 ? a.h : a.h2;
    const W = capa === 1 ? a.w : a.w2;
    const F = capa === 1 ? a.f1 : a.f2;
    const wOff = capa === 1 ? a.oW1 : a.oW2;
    const bOff = capa === 1 ? a.oB1 : a.oB2;
    const z = new Float32Array(F * H * W);
    const salida = new Float32Array(F * (H >> 1) * (W >> 1));
    cubos.convForward(entrada, C, H, W, p, wOff, bOff, fIni, fFin, z);
    cubos.reluPool(z, fIni, fFin, H, W, salida, null);
    const porFiltro = (H >> 1) * (W >> 1);
    return salida.slice(fIni * porFiltro, fFin * porFiltro);
}

function mlpMicro(a, p, capa, jIni, jFin, entrada) {
    const nIn = capa === 1 ? a.plano : a.oculta;
    const nOut = capa === 1 ? a.oculta : a.k;
    const salida = new Float32Array(nOut);
    if (capa === 1) {
        cubos.densaForward(entrada, nIn, p, a.oW3, a.oB3, jIni, jFin, salida, true);
    } else {
        cubos.densaForward(entrada, nIn, p, a.oW4, a.oB4, jIni, jFin, salida, false);
    }
    return salida.slice(jIni, jFin);
}

// Dimensiones (C, H, W, F) que espera CONV_FWD en cada capa y (nIn, nOut) de MLP_FWD.
function dimsConv(a, capa) {
    return capa === 1 ? [a.c, a.h, a.w, a.f1] : [a.f1, a.h2, a.w2, a.f2];
}

function dimsMlp(a, capa) {
    return capa === 1 ? [a.plano, a.oculta] : [a.oculta, a.k];
}

module.exports = {
    Arquitectura, nuevoEspacio, normalizar, predecir, entrenarMuestra, entrenarLote, inferirLote,
    convMicro, mlpMicro, dimsConv, dimsMlp,
};
