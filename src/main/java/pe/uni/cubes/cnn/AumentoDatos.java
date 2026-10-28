package pe.uni.cubes.cnn;

import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.util.Locale;
import java.util.Random;

/**
 * Data augmentation sobre tensores CHW de bytes (los mismos que viajan en TAREA_ENTRENAR).
 * Cada llamada sortea transformaciones nuevas: volteo horizontal, recorte aleatorio con
 * relleno (desplazamiento), rotacion pequena, brillo/contraste y ruido gaussiano.
 * Las transformaciones geometricas se combinan en un solo AffineTransform y se muestrean
 * con interpolacion bilineal, asi la imagen se cuantiza a bytes una sola vez.
 */
public final class AumentoDatos {

    public enum Relleno { REFLEJO, BORDE, GRIS }

    private final int c, h, w;
    private final double probVolteo;
    private final int desplazamiento;
    private final Relleno relleno;
    private final double grados, probRotacion;
    private final double probColor, brillo, contraste;
    private final double probRuido, sigmaRuido;
    private final double[] puntos;
    private final float[] buf;

    public AumentoDatos(int c, int h, int w, double probVolteo, int desplazamiento, Relleno relleno,
                        double grados, double probRotacion, double probColor, double brillo, double contraste,
                        double probRuido, double sigmaRuido) {
        this.c = c;
        this.h = h;
        this.w = w;
        this.probVolteo = probVolteo;
        this.desplazamiento = Math.max(0, desplazamiento);
        this.relleno = relleno;
        this.grados = grados;
        this.probRotacion = probRotacion;
        this.probColor = probColor;
        this.brillo = brillo;
        this.contraste = contraste;
        this.probRuido = probRuido;
        this.sigmaRuido = sigmaRuido;
        puntos = new double[2 * h * w];
        buf = new float[c * h * w];
    }

    /** Escribe en dst (desde dstOff) una version aumentada de la imagen src (desde srcOff). No es thread-safe. */
    public void aplicar(byte[] src, int srcOff, byte[] dst, int dstOff, Random r) {
        int hw = h * w, n = c * hw;
        boolean volteo = r.nextDouble() < probVolteo;
        int dx = desplazamiento > 0 ? r.nextInt(2 * desplazamiento + 1) - desplazamiento : 0;
        int dy = desplazamiento > 0 ? r.nextInt(2 * desplazamiento + 1) - desplazamiento : 0;
        double angulo = grados > 0 && r.nextDouble() < probRotacion ? (2 * r.nextDouble() - 1) * grados : 0;

        if (volteo || dx != 0 || dy != 0 || angulo != 0) {
            AffineTransform t = new AffineTransform();
            t.translate(w / 2.0 + dx, h / 2.0 + dy);
            t.rotate(Math.toRadians(angulo));
            if (volteo) {
                t.scale(-1, 1);
            }
            t.translate(-w / 2.0, -h / 2.0);
            for (int y = 0, k = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    puntos[k++] = x + 0.5;
                    puntos[k++] = y + 0.5;
                }
            }
            try {
                t.createInverse().transform(puntos, 0, puntos, 0, hw);
            } catch (NoninvertibleTransformException ex) {
                throw new IllegalStateException(ex);
            }
            for (int ch = 0; ch < c; ch++) {
                int base = srcOff + ch * hw;
                for (int i = 0; i < hw; i++) {
                    buf[ch * hw + i] = muestrear(src, base, puntos[2 * i] - 0.5, puntos[2 * i + 1] - 0.5);
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                buf[i] = src[srcOff + i] & 0xFF;
            }
        }

        if (r.nextDouble() < probColor) {
            double media = 0;
            for (int i = 0; i < n; i++) {
                media += buf[i];
            }
            media /= n;
            double fc = 1 + (2 * r.nextDouble() - 1) * contraste;
            double fb = (2 * r.nextDouble() - 1) * brillo * 255;
            for (int i = 0; i < n; i++) {
                buf[i] = (float) ((buf[i] - media) * fc + media + fb);
            }
        }
        if (r.nextDouble() < probRuido) {
            double s = sigmaRuido * 255;
            for (int i = 0; i < n; i++) {
                buf[i] += (float) (r.nextGaussian() * s);
            }
        }
        for (int i = 0; i < n; i++) {
            int v = Math.round(buf[i]);
            dst[dstOff + i] = (byte) (v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }

    private float muestrear(byte[] src, int base, double sx, double sy) {
        int x0 = (int) Math.floor(sx), y0 = (int) Math.floor(sy);
        double fx = sx - x0, fy = sy - y0;
        double v00 = pixel(src, base, x0, y0), v10 = pixel(src, base, x0 + 1, y0);
        double v01 = pixel(src, base, x0, y0 + 1), v11 = pixel(src, base, x0 + 1, y0 + 1);
        return (float) ((v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy);
    }

    private int pixel(byte[] src, int base, int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) {
            if (relleno == Relleno.GRIS) {
                return 128;
            }
            x = relleno == Relleno.REFLEJO ? reflejar(x, w) : Math.max(0, Math.min(w - 1, x));
            y = relleno == Relleno.REFLEJO ? reflejar(y, h) : Math.max(0, Math.min(h - 1, y));
        }
        return src[base + y * w + x] & 0xFF;
    }

    private static int reflejar(int i, int n) {
        while (i < 0 || i >= n) {
            i = i < 0 ? -i - 1 : 2 * n - 1 - i;
        }
        return i;
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "volteo p=%.2f | recorte +-%d px (relleno %s) | rotacion +-%.0f grados p=%.2f"
                        + " | brillo +-%.2f, contraste +-%.2f p=%.2f | ruido sigma %.3f p=%.2f",
                probVolteo, desplazamiento, relleno.name().toLowerCase(), grados, probRotacion,
                brillo, contraste, probColor, sigmaRuido, probRuido);
    }
}
