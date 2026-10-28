package pe.uni.cubes.cnn;

import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;

public final class Arquitectura {

    public final int c, h, w, f1, f2, oculta, k;
    public final int h2, w2, h4, w4, plano;
    public final int oW1, oB1, oW2, oB2, oW3, oB3, oW4, oB4, totalParametros;

    public Arquitectura(int c, int h, int w, int f1, int f2, int oculta, int k) {
        if (h % 4 != 0 || w % 4 != 0) {
            throw new IllegalArgumentException("H y W deben ser multiplos de 4");
        }
        this.c = c;
        this.h = h;
        this.w = w;
        this.f1 = f1;
        this.f2 = f2;
        this.oculta = oculta;
        this.k = k;
        h2 = h / 2;
        w2 = w / 2;
        h4 = h / 4;
        w4 = w / 4;
        plano = f2 * h4 * w4;
        oW1 = 0;
        oB1 = oW1 + f1 * c * 9;
        oW2 = oB1 + f1;
        oB2 = oW2 + f2 * f1 * 9;
        oW3 = oB2 + f2;
        oB3 = oW3 + oculta * plano;
        oW4 = oB3 + oculta;
        oB4 = oW4 + k * oculta;
        totalParametros = oB4 + k;
    }

    public int tamImagen() {
        return c * h * w;
    }

    public void escribir(Escritor e) {
        e.i32(c).i32(h).i32(w).i32(f1).i32(f2).i32(oculta).i32(k);
    }

    public static Arquitectura leer(Lector l) {
        return new Arquitectura(l.i32(), l.i32(), l.i32(), l.i32(), l.i32(), l.i32(), l.i32());
    }

    public boolean igual(Arquitectura o) {
        return o != null && c == o.c && h == o.h && w == o.w && f1 == o.f1 && f2 == o.f2 && oculta == o.oculta && k == o.k;
    }

    @Override
    public String toString() {
        return String.format("CNN %dx%dx%d | conv%d -> pool | conv%d -> pool | densa %d->%d | densa %d->%d | %d parametros",
                c, h, w, f1, f2, plano, oculta, oculta, k, totalParametros);
    }
}
