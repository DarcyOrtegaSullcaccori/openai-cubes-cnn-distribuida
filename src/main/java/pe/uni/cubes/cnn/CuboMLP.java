package pe.uni.cubes.cnn;

public final class CuboMLP {

    private CuboMLP() {
    }

    public static void forward(float[] x, int nIn, float[] p, int wOff, int bOff,
                               int jIni, int jFin, float[] out, boolean relu) {
        for (int j = jIni; j < jFin; j++) {
            int wo = wOff + j * nIn;
            float s = 0f;
            for (int i = 0; i < nIn; i++) {
                s += p[wo + i] * x[i];
            }
            s += p[bOff + j];
            out[j] = (relu && s < 0f) ? 0f : s;
        }
    }

    public static void backward(float[] x, int nIn, float[] dOut, int nOut, float[] p, int wOff, int bOff,
                                float[] g, float[] dIn) {
        for (int j = 0; j < nOut; j++) {
            float d = dOut[j];
            if (d == 0f) {
                continue;
            }
            g[bOff + j] += d;
            int wo = wOff + j * nIn;
            for (int i = 0; i < nIn; i++) {
                g[wo + i] += d * x[i];
            }
            if (dIn != null) {
                for (int i = 0; i < nIn; i++) {
                    dIn[i] += p[wo + i] * d;
                }
            }
        }
    }

    public static void softmax(float[] z, int n, float[] prob) {
        float max = z[0];
        for (int i = 1; i < n; i++) {
            if (z[i] > max) {
                max = z[i];
            }
        }
        double suma = 0;
        for (int i = 0; i < n; i++) {
            double e = Math.exp(z[i] - max);
            prob[i] = (float) e;
            suma += e;
        }
        for (int i = 0; i < n; i++) {
            prob[i] = (float) (prob[i] / suma);
        }
    }
}
