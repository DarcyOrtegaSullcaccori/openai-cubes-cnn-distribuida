package pe.uni.cubes.cnn;

/**
 * Test-time augmentation (TTA): genera varias vistas de cada imagen (original, espejo y
 * desplazamientos de 2 px con relleno reflejado) para promediar las probabilidades del
 * modelo sobre ellas. 1 vista = sin TTA, 2 = original + espejo, 10 = ambas con 5 desplazamientos.
 * Las vistas son imagenes normales, asi que viajan en TAREA_INFERIR sin cambiar el protocolo.
 */
public final class VistasTTA {

    private static final int[][] DESPLAZAMIENTOS = {{0, 0}, {2, 0}, {-2, 0}, {0, 2}, {0, -2}};

    private VistasTTA() {
    }

    /** Lleva cualquier valor a una cantidad de vistas soportada: 1, 2 o 10. */
    public static int normalizar(int vistas) {
        return vistas >= 10 ? 10 : (vistas >= 2 ? 2 : 1);
    }

    /** Devuelve n*vistas imagenes CHW; las vistas de cada imagen quedan contiguas. */
    public static byte[] generar(byte[] imgs, int off, int n, int c, int h, int w, int vistas) {
        int hw = h * w, tam = c * hw;
        byte[] d = new byte[n * vistas * tam];
        for (int m = 0; m < n; m++) {
            int src = off + m * tam;
            for (int v = 0; v < vistas; v++) {
                boolean volteo = (v & 1) == 1;
                int dx = DESPLAZAMIENTOS[v >> 1][0], dy = DESPLAZAMIENTOS[v >> 1][1];
                int dst = (m * vistas + v) * tam;
                for (int ch = 0; ch < c; ch++) {
                    for (int y = 0; y < h; y++) {
                        int sy = reflejar(y - dy, h);
                        for (int x = 0; x < w; x++) {
                            int sx = reflejar((volteo ? w - 1 - x : x) - dx, w);
                            d[dst + ch * hw + y * w + x] = imgs[src + ch * hw + sy * w + sx];
                        }
                    }
                }
            }
        }
        return d;
    }

    /** Promedia las probabilidades de las vistas contiguas: de n*vistas filas a n filas de k clases. */
    public static float[] promediar(float[] probs, int n, int k, int vistas) {
        float[] r = new float[n * k];
        for (int m = 0; m < n; m++) {
            for (int j = 0; j < k; j++) {
                float s = 0;
                for (int v = 0; v < vistas; v++) {
                    s += probs[(m * vistas + v) * k + j];
                }
                r[m * k + j] = s / vistas;
            }
        }
        return r;
    }

    private static int reflejar(int i, int n) {
        return i < 0 ? -i - 1 : (i >= n ? 2 * n - 1 - i : i);
    }
}
