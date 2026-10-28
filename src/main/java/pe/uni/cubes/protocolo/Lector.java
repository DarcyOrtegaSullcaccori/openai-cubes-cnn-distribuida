package pe.uni.cubes.protocolo;

import java.nio.Buffer;
import java.nio.ByteBuffer;

public final class Lector {

    private final ByteBuffer bb;

    public Lector(byte[] payload) {
        bb = ByteBuffer.wrap(payload);
    }

    public int i32() {
        return bb.getInt();
    }

    public float f32() {
        return bb.getFloat();
    }

    public String str() {
        int n = bb.getInt();
        byte[] b = new byte[n];
        bb.get(b);
        return new String(b, Escritor.UTF8);
    }

    public int[] i32s(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = bb.getInt();
        }
        return a;
    }

    public float[] f32s(int n) {
        float[] a = new float[n];
        f32s(a, 0, n);
        return a;
    }

    public void f32s(float[] dst, int off, int n) {
        bb.asFloatBuffer().get(dst, off, n);
        ((Buffer) bb).position(bb.position() + n * 4);
    }

    public byte[] bytes(int n) {
        byte[] b = new byte[n];
        bb.get(b);
        return b;
    }

    public byte[] blob() {
        return bytes(bb.getInt());
    }

    public int restantes() {
        return bb.remaining();
    }
}
