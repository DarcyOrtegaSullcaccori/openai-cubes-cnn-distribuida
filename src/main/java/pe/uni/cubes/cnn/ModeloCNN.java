package pe.uni.cubes.cnn;

import java.util.Random;

public final class ModeloCNN {

    public final Arquitectura a;

    public ModeloCNN(Arquitectura a) {
        this.a = a;
    }

    public static final class Espacio {
        final float[] x, z1, a1, z2, a2, z3, a3, o, prob;
        final int[] i1, i2;
        final float[] dO, dA3, dZ3, dA2, dZ2, dA1, dZ1;
        public float perdida;
        public boolean acierto;

        Espacio(Arquitectura a) {
            x = new float[a.c * a.h * a.w];
            z1 = new float[a.f1 * a.h * a.w];
            a1 = new float[a.f1 * a.h2 * a.w2];
            i1 = new int[a1.length];
            z2 = new float[a.f2 * a.h2 * a.w2];
            a2 = new float[a.plano];
            i2 = new int[a2.length];
            z3 = new float[a.oculta];
            a3 = new float[a.oculta];
            o = new float[a.k];
            prob = new float[a.k];
            dO = new float[a.k];
            dA3 = new float[a.oculta];
            dZ3 = new float[a.oculta];
            dA2 = new float[a.plano];
            dZ2 = new float[z2.length];
            dA1 = new float[a1.length];
            dZ1 = new float[z1.length];
        }
    }

    public Espacio nuevoEspacio() {
        return new Espacio(a);
    }

    public static float[] inicializar(Arquitectura a, long semilla) {
        Random r = new Random(semilla);
        float[] p = new float[a.totalParametros];
        llenarHe(p, a.oW1, a.f1 * a.c * 9, a.c * 9, r);
        llenarHe(p, a.oW2, a.f2 * a.f1 * 9, a.f1 * 9, r);
        llenarHe(p, a.oW3, a.oculta * a.plano, a.plano, r);
        llenarHe(p, a.oW4, a.k * a.oculta, a.oculta, r);
        return p;
    }

    private static void llenarHe(float[] p, int off, int n, int fanIn, Random r) {
        double std = Math.sqrt(2.0 / fanIn);
        for (int i = 0; i < n; i++) {
            p[off + i] = (float) (r.nextGaussian() * std);
        }
    }

    public static void normalizar(byte[] imgs, int off, float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = (imgs[off + i] & 0xFF) / 255f - 0.5f;
        }
    }

    private void forward(float[] p, Espacio e) {
        CuboConvolucion.forward(e.x, a.c, a.h, a.w, p, a.oW1, a.oB1, 0, a.f1, e.z1);
        CuboConvolucion.reluPool(e.z1, 0, a.f1, a.h, a.w, e.a1, e.i1);
        CuboConvolucion.forward(e.a1, a.f1, a.h2, a.w2, p, a.oW2, a.oB2, 0, a.f2, e.z2);
        CuboConvolucion.reluPool(e.z2, 0, a.f2, a.h2, a.w2, e.a2, e.i2);
        CuboMLP.forward(e.a2, a.plano, p, a.oW3, a.oB3, 0, a.oculta, e.z3, false);
        for (int j = 0; j < a.oculta; j++) {
            e.a3[j] = e.z3[j] > 0f ? e.z3[j] : 0f;
        }
        CuboMLP.forward(e.a3, a.oculta, p, a.oW4, a.oB4, 0, a.k, e.o, false);
        CuboMLP.softmax(e.o, a.k, e.prob);
    }

    public void predecir(float[] p, byte[] imgs, int off, float[] probs, int pOff, Espacio e) {
        normalizar(imgs, off, e.x);
        forward(p, e);
        System.arraycopy(e.prob, 0, probs, pOff, a.k);
    }

    public float entrenarMuestra(float[] p, byte[] imgs, int off, int etiqueta, float[] g, Espacio e) {
        normalizar(imgs, off, e.x);
        forward(p, e);
        int mejor = 0;
        for (int i = 1; i < a.k; i++) {
            if (e.prob[i] > e.prob[mejor]) {
                mejor = i;
            }
        }
        e.acierto = mejor == etiqueta;
        e.perdida = (float) -Math.log(e.prob[etiqueta] + 1e-9);
        for (int i = 0; i < a.k; i++) {
            e.dO[i] = e.prob[i] - (i == etiqueta ? 1f : 0f);
        }
        java.util.Arrays.fill(e.dA3, 0f);
        CuboMLP.backward(e.a3, a.oculta, e.dO, a.k, p, a.oW4, a.oB4, g, e.dA3);
        for (int j = 0; j < a.oculta; j++) {
            e.dZ3[j] = e.z3[j] > 0f ? e.dA3[j] : 0f;
        }
        java.util.Arrays.fill(e.dA2, 0f);
        CuboMLP.backward(e.a2, a.plano, e.dZ3, a.oculta, p, a.oW3, a.oB3, g, e.dA2);
        CuboConvolucion.reluPoolBackward(e.dA2, e.i2, e.z2, a.f2, a.h2, a.w2, e.dZ2);
        java.util.Arrays.fill(e.dA1, 0f);
        CuboConvolucion.backward(e.a1, a.f1, a.h2, a.w2, e.dZ2, 0, a.f2, p, a.oW2, a.oB2, g, e.dA1);
        CuboConvolucion.reluPoolBackward(e.dA1, e.i1, e.z1, a.f1, a.h, a.w, e.dZ1);
        CuboConvolucion.backward(e.x, a.c, a.h, a.w, e.dZ1, 0, a.f1, p, a.oW1, a.oB1, g, null);
        return e.perdida;
    }
}
