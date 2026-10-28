package pe.uni.cubes.protocolo;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

public final class Escritor {

    public static final Charset UTF8 = Charset.forName("UTF-8");

    private final ByteArrayOutputStream bos;

    public Escritor() {
        this(256);
    }

    public Escritor(int capacidad) {
        bos = new ByteArrayOutputStream(capacidad);
    }

    public Escritor i32(int v) {
        bos.write((v >>> 24) & 0xFF);
        bos.write((v >>> 16) & 0xFF);
        bos.write((v >>> 8) & 0xFF);
        bos.write(v & 0xFF);
        return this;
    }

    public Escritor f32(float v) {
        return i32(Float.floatToIntBits(v));
    }

    public Escritor str(String s) {
        byte[] b = (s == null ? "" : s).getBytes(UTF8);
        i32(b.length);
        bos.write(b, 0, b.length);
        return this;
    }

    public Escritor i32s(int[] a, int off, int n) {
        for (int i = 0; i < n; i++) {
            i32(a[off + i]);
        }
        return this;
    }

    public Escritor f32s(float[] a, int off, int n) {
        ByteBuffer bb = ByteBuffer.allocate(n * 4);
        bb.asFloatBuffer().put(a, off, n);
        bos.write(bb.array(), 0, n * 4);
        return this;
    }

    public Escritor f32s(float[] a) {
        return f32s(a, 0, a.length);
    }

    public Escritor bytes(byte[] a, int off, int n) {
        bos.write(a, off, n);
        return this;
    }

    public Escritor blob(byte[] a) {
        i32(a.length);
        bos.write(a, 0, a.length);
        return this;
    }

    public byte[] toBytes() {
        return bos.toByteArray();
    }
}
