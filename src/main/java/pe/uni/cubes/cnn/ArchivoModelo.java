package pe.uni.cubes.cnn;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;

public final class ArchivoModelo {

    private static final int MAGIC = 0x4F435731;

    public Arquitectura arquitectura;
    public List<String> clases = new ArrayList<String>();
    public float[] pesos;
    public int epocas;

    public void guardar(File f) throws IOException {
        File dir = f.getAbsoluteFile().getParentFile();
        if (dir != null && !dir.exists()) {
            dir.mkdirs();
        }
        Escritor e = new Escritor(pesos.length * 4 + 1024);
        e.i32(MAGIC);
        arquitectura.escribir(e);
        e.i32(clases.size());
        for (String c : clases) {
            e.str(c);
        }
        e.i32(epocas);
        e.i32(pesos.length);
        e.f32s(pesos);
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)));
        try {
            out.write(e.toBytes());
        } finally {
            out.close();
        }
    }

    public static ArchivoModelo cargar(File f) throws IOException {
        byte[] datos = new byte[(int) f.length()];
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)));
        try {
            in.readFully(datos);
        } finally {
            in.close();
        }
        Lector l = new Lector(datos);
        if (l.i32() != MAGIC) {
            throw new IOException("No es un archivo de modelo .ocw valido");
        }
        ArchivoModelo m = new ArchivoModelo();
        m.arquitectura = Arquitectura.leer(l);
        int nc = l.i32();
        for (int i = 0; i < nc; i++) {
            m.clases.add(l.str());
        }
        m.epocas = l.i32();
        int p = l.i32();
        if (p != m.arquitectura.totalParametros) {
            throw new IOException("El archivo no coincide con su arquitectura");
        }
        m.pesos = l.f32s(p);
        return m;
    }
}
