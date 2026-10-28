package pe.uni.cubes.cnn;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Dataset {

    public static final String[] CIFAR10 = {"airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"};

    public final List<String> clases;
    public final int tam;
    public byte[] imgs;
    public int[] etiquetas;
    public int n;

    private Dataset(List<String> clases, int tam, byte[] imgs, int[] etiquetas, int n) {
        this.clases = clases;
        this.tam = tam;
        this.imgs = imgs;
        this.etiquetas = etiquetas;
        this.n = n;
    }

    public int[] conteoPorClase() {
        int[] c = new int[clases.size()];
        for (int i = 0; i < n; i++) {
            c[etiquetas[i]]++;
        }
        return c;
    }

    public void barajar(long semilla) {
        java.util.Random r = new java.util.Random(semilla);
        byte[] tmp = new byte[tam];
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            System.arraycopy(imgs, i * tam, tmp, 0, tam);
            System.arraycopy(imgs, j * tam, imgs, i * tam, tam);
            System.arraycopy(tmp, 0, imgs, j * tam, tam);
            int e = etiquetas[i];
            etiquetas[i] = etiquetas[j];
            etiquetas[j] = e;
        }
    }

    public String resumen() {
        StringBuilder sb = new StringBuilder();
        int[] c = conteoPorClase();
        for (int i = 0; i < c.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(clases.get(i)).append('=').append(c[i]);
        }
        return n + " imagenes [" + sb + "]";
    }

    private static final class Acum {
        final ByteArrayOutputStream bytes = new ByteArrayOutputStream(1 << 20);
        final List<Integer> etq = new ArrayList<Integer>();
        final int[] porClase;

        Acum(int k) {
            porClase = new int[k];
        }

        boolean agregar(byte[] img, int off, int tam, int clase, int max) {
            if (max > 0 && porClase[clase] >= max) {
                return false;
            }
            bytes.write(img, off, tam);
            etq.add(clase);
            porClase[clase]++;
            return true;
        }

        Dataset construir(List<String> clases, int tam) {
            int[] e = new int[etq.size()];
            for (int i = 0; i < e.length; i++) {
                e[i] = etq.get(i);
            }
            return new Dataset(clases, tam, bytes.toByteArray(), e, e.length);
        }
    }

    public static Dataset[] cargar(String cifarDir, String cifarClases, String carpeta, int maxPorClase,
                                   List<String> clasesFijas, int w, int h, StringBuilder log) throws IOException {
        int tam = 3 * w * h;
        Map<Integer, String> mapaCifar = new HashMap<Integer, String>();
        List<String> ordenCifar = new ArrayList<String>();
        File dirCifar = new File(cifarDir == null ? "" : cifarDir);
        if (cifarClases != null && cifarClases.trim().length() > 0 && dirCifar.isDirectory()) {
            if (w != 32 || h != 32) {
                throw new IOException("CIFAR-10 requiere entrada 32x32");
            }
            for (String par : cifarClases.split(",")) {
                String[] kv = par.trim().split(":");
                int idx = Arrays.asList(CIFAR10).indexOf(kv[0].trim());
                if (idx < 0) {
                    throw new IOException("Clase CIFAR-10 desconocida: " + kv[0]);
                }
                String nombre = kv.length > 1 ? kv[1].trim() : kv[0].trim();
                mapaCifar.put(idx, nombre);
                if (!ordenCifar.contains(nombre)) {
                    ordenCifar.add(nombre);
                }
            }
        }
        File raiz = new File(carpeta == null ? "" : carpeta);
        File dirTrain = new File(raiz, "train");
        File dirTest = new File(raiz, "test");
        if (!dirTrain.isDirectory() && raiz.isDirectory()) {
            dirTrain = raiz;
            dirTest = null;
        }
        List<String> clasesCarpeta = new ArrayList<String>();
        if (dirTrain.isDirectory()) {
            File[] subs = dirTrain.listFiles();
            if (subs != null) {
                Arrays.sort(subs);
                for (File s : subs) {
                    if (s.isDirectory() && !s.getName().equals("test") && !s.getName().startsWith(".")) {
                        clasesCarpeta.add(s.getName());
                    }
                }
            }
        }
        List<String> clases;
        if (clasesFijas != null) {
            clases = new ArrayList<String>(clasesFijas);
        } else {
            clases = new ArrayList<String>(ordenCifar);
            for (String c : clasesCarpeta) {
                if (!clases.contains(c)) {
                    clases.add(c);
                }
            }
        }
        if (clases.isEmpty()) {
            throw new IOException("No hay datos: coloque CIFAR-10 en '" + cifarDir + "' o imagenes en '" + carpeta
                    + "/train/<clase>/'. Tambien puede ejecutar la herramienta GenerarSinteticos.");
        }
        Acum train = new Acum(clases.size());
        Acum test = new Acum(clases.size());
        if (!mapaCifar.isEmpty()) {
            for (int b = 1; b <= 5; b++) {
                leerCifar(new File(dirCifar, "data_batch_" + b + ".bin"), mapaCifar, clases, train, maxPorClase, log);
            }
            leerCifar(new File(dirCifar, "test_batch.bin"), mapaCifar, clases, test, 0, log);
        }
        boolean separar = dirTest == null || !dirTest.isDirectory();
        for (String c : clasesCarpeta) {
            int idx = clases.indexOf(c);
            if (idx < 0) {
                log.append("  (se ignora la carpeta '").append(c).append("': no esta en las clases del modelo)\n");
                continue;
            }
            leerCarpeta(new File(dirTrain, c), idx, train, separar ? test : null, maxPorClase, w, h, log);
            if (!separar) {
                leerCarpeta(new File(dirTest, c), idx, test, null, 0, w, h, log);
            }
        }
        Dataset dTrain = train.construir(clases, tam);
        Dataset dTest = test.construir(clases, tam);
        dTrain.barajar(11);
        dTest.barajar(13);
        log.append("  Entrenamiento: ").append(dTrain.resumen()).append('\n');
        log.append("  Prueba:        ").append(dTest.resumen()).append('\n');
        return new Dataset[]{dTrain, dTest};
    }

    private static void leerCifar(File f, Map<Integer, String> mapa, List<String> clases, Acum acum, int max,
                                  StringBuilder log) throws IOException {
        if (!f.isFile()) {
            return;
        }
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f), 1 << 16));
        byte[] reg = new byte[3072];
        int usados = 0;
        try {
            while (true) {
                int etq;
                try {
                    etq = in.readUnsignedByte();
                } catch (EOFException eof) {
                    break;
                }
                in.readFully(reg);
                String nombre = mapa.get(etq);
                if (nombre != null && acum.agregar(reg, 0, 3072, clases.indexOf(nombre), max)) {
                    usados++;
                }
            }
        } finally {
            in.close();
        }
        log.append("  CIFAR ").append(f.getName()).append(": ").append(usados).append(" imagenes\n");
    }

    private static void leerCarpeta(File dir, int clase, Acum acum, Acum testSeparado, int max, int w, int h,
                                    StringBuilder log) {
        File[] archivos = dir.listFiles();
        if (archivos == null) {
            return;
        }
        Arrays.sort(archivos);
        int usados = 0, i = 0;
        for (File a : archivos) {
            if (!a.isFile() || !ImagenUtil.esImagen(a.getName())) {
                continue;
            }
            try {
                byte[] t = ImagenUtil.aTensor(ImagenUtil.leer(a), w, h);
                if (testSeparado != null && i % 10 == 9) {
                    testSeparado.agregar(t, 0, t.length, clase, 0);
                } else if (acum.agregar(t, 0, t.length, clase, max)) {
                    usados++;
                }
                i++;
            } catch (IOException ex) {
                log.append("  (imagen ignorada ").append(a.getName()).append(": ").append(ex.getMessage()).append(")\n");
            }
        }
        log.append("  Carpeta ").append(dir.getPath()).append(": ").append(usados + (testSeparado != null ? i - usados : 0))
                .append(" imagenes\n");
    }
}
