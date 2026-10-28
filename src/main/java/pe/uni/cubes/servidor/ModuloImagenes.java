package pe.uni.cubes.servidor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import pe.uni.cubes.cnn.ArchivoModelo;
import pe.uni.cubes.cnn.Arquitectura;
import pe.uni.cubes.cnn.AumentoDatos;
import pe.uni.cubes.cnn.CuboMLP;
import pe.uni.cubes.cnn.Dataset;
import pe.uni.cubes.cnn.ModeloCNN;
import pe.uni.cubes.cnn.MotorCNN;
import pe.uni.cubes.cnn.VistasTTA;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class ModuloImagenes {

    public static final class Prediccion {
        public float[] probs;
        public String trabajador;
        public int msCalculo;
    }

    private static final class Paso {
        float[] grad;
        int n;
        float perdida;
        int aciertos;
    }

    private final Configuracion cfg;
    private final RegistroNodos registro;
    private final MotorCNN motorLocal;
    private final long timeoutMs;

    private volatile Arquitectura arq;
    private volatile float[] pesos;
    private float[] momento;
    private volatile int version;
    private volatile List<String> clases = new ArrayList<String>();
    private int epocasTotales;
    private Dataset train, test;

    private final Object candadoPayload = new Object();
    private byte[] payloadSync;
    private int versionPayloadSync = -1;
    private volatile String modoInferencia;
    private volatile int vistasTta;
    private volatile boolean entrenando;

    public ModuloImagenes(Configuracion cfg, RegistroNodos registro) {
        this.cfg = cfg;
        this.registro = registro;
        this.motorLocal = new MotorCNN(cfg.ent("hilos.locales", Runtime.getRuntime().availableProcessors()));
        this.timeoutMs = cfg.ent("tarea.timeoutMs", 120000);
        this.modoInferencia = cfg.txt("inferencia.modo", "datos");
        this.vistasTta = VistasTTA.normalizar(cfg.ent("inferencia.tta", 1));
        registro.setAlRegistrar(n -> {
            if (pesos != null) {
                try {
                    sincronizar(n);
                } catch (Exception ex) {
                    System.out.println("[Modelo] No se pudo sincronizar " + n + ": " + ex.getMessage());
                }
            }
        });
    }

    public void cargarModelo() {
        File f = new File(cfg.txt("modelo.archivo", "modelos/modelo.ocw"));
        if (!f.isFile()) {
            System.out.println("[Modelo] No hay modelo entrenado aun (" + f.getPath() + "). Use 'entrenar'.");
            return;
        }
        try {
            ArchivoModelo m = ArchivoModelo.cargar(f);
            arq = m.arquitectura;
            pesos = m.pesos;
            momento = new float[pesos.length];
            clases = m.clases;
            epocasTotales = m.epocas;
            version++;
            System.out.println("[Modelo] Cargado " + f.getPath() + " | clases " + clases + " | epocas " + epocasTotales);
            System.out.println("[Modelo] " + arq);
        } catch (IOException ex) {
            System.out.println("[Modelo] Error al cargar: " + ex.getMessage());
        }
    }

    public void guardarModelo() throws IOException {
        if (pesos == null) {
            System.out.println("[Modelo] No hay modelo para guardar.");
            return;
        }
        ArchivoModelo m = new ArchivoModelo();
        m.arquitectura = arq;
        m.clases = clases;
        m.pesos = pesos;
        m.epocas = epocasTotales;
        File f = new File(cfg.txt("modelo.archivo", "modelos/modelo.ocw"));
        m.guardar(f);
        System.out.println("[Modelo] Guardado en " + f.getPath());
    }

    public void reiniciar() {
        pesos = null;
        train = null;
        test = null;
        clases = new ArrayList<String>();
        epocasTotales = 0;
        System.out.println("[Modelo] Modelo reiniciado: el proximo 'entrenar' parte de pesos aleatorios.");
    }

    public boolean tieneModelo() {
        return pesos != null;
    }

    public List<String> getClases() {
        return clases;
    }

    public Arquitectura getArquitectura() {
        return arq;
    }

    public String getModo() {
        return modoInferencia;
    }

    public void setModo(String m) {
        modoInferencia = m;
    }

    public int getVistasTta() {
        return vistasTta;
    }

    public void setVistasTta(int v) {
        vistasTta = VistasTTA.normalizar(v);
    }

    public boolean isEntrenando() {
        return entrenando;
    }

    private void cargarDatos() throws IOException {
        if (train != null) {
            return;
        }
        System.out.println("[Datos] Cargando datos...");
        StringBuilder log = new StringBuilder();
        Dataset[] d = Dataset.cargar(cfg.txt("dataset.cifar.dir", "datos/cifar-10-batches-bin"),
                cfg.txt("dataset.cifar.clases", ""), cfg.txt("dataset.carpeta", "datos/imagenes"),
                cfg.ent("dataset.maxPorClase", 0), pesos != null ? clases : null, 32, 32, log);
        System.out.print(log);
        train = d[0];
        test = d[1];
        if (pesos == null) {
            clases = train.clases;
            arq = new Arquitectura(3, 32, 32, cfg.ent("cnn.filtros1", 8), cfg.ent("cnn.filtros2", 16),
                    cfg.ent("cnn.oculta", 64), clases.size());
            pesos = ModeloCNN.inicializar(arq, cfg.ent("cnn.semilla", 42));
            momento = new float[pesos.length];
            version++;
            System.out.println("[Modelo] Nuevo modelo: " + arq);
            System.out.println("[Modelo] Clases: " + clases);
        }
    }

    private byte[] payloadSync() {
        synchronized (candadoPayload) {
            if (versionPayloadSync != version) {
                float[] p = pesos;
                Escritor e = new Escritor(p.length * 4 + 64).i32(version);
                arq.escribir(e);
                e.i32(p.length).f32s(p);
                payloadSync = e.toBytes();
                versionPayloadSync = version;
            }
            return payloadSync;
        }
    }

    public void sincronizar(ConexionNodo n) throws Exception {
        synchronized (n) {
            int v = version;
            if (n.versionPesos == v) {
                return;
            }
            byte[] payload = payloadSync();
            Trama ack = n.solicitar(Protocolo.SYNC_PESOS, payload, timeoutMs);
            n.versionPesos = ack.lector().i32();
        }
    }

    private void sincronizarTodos(List<ConexionNodo> ns) {
        List<Thread> hilos = new ArrayList<Thread>();
        for (final ConexionNodo n : ns) {
            if (n.versionPesos == version) {
                continue;
            }
            Thread t = new Thread(() -> {
                try {
                    sincronizar(n);
                } catch (Exception ex) {
                    System.out.println("[Modelo] Fallo SYNC con " + n + ": " + ex.getMessage());
                    n.cerrar();
                }
            });
            t.start();
            hilos.add(t);
        }
        for (Thread t : hilos) {
            try {
                t.join();
            } catch (InterruptedException ignored) {
            }
        }
    }

    private Paso pasoLocal(MotorCNN motor, byte[] imgs, int[] etq, int n) throws Exception {
        MotorCNN.ResultadoEntrenamiento r = motor.entrenar(arq, pesos, imgs, etq, n);
        Paso p = new Paso();
        p.grad = r.gradiente;
        p.n = r.n;
        p.perdida = r.sumaPerdida;
        p.aciertos = r.aciertos;
        return p;
    }

    static int[] repartir(int b, List<ConexionNodo> ns) {
        int[] cuota = new int[ns.size()];
        for (int m = 0; m < b; m++) {
            int mejor = 0;
            double mejorFin = Double.MAX_VALUE;
            for (int i = 0; i < ns.size(); i++) {
                double fin = (cuota[i] + 1) / Math.max(ns.get(i).velocidad, 1e-6);
                if (fin < mejorFin) {
                    mejorFin = fin;
                    mejor = i;
                }
            }
            cuota[mejor]++;
        }
        return cuota;
    }

    private Paso pasoDistribuido(byte[] imgs, int[] etq, int b) throws Exception {
        List<ConexionNodo> ns = registro.vivos();
        if (ns.isEmpty()) {
            return pasoLocal(motorLocal, imgs, etq, b);
        }
        sincronizarTodos(ns);
        ns = registro.vivos();
        if (ns.isEmpty()) {
            return pasoLocal(motorLocal, imgs, etq, b);
        }
        int tam = arq.tamImagen();
        int[] cuotas = repartir(b, ns);
        List<CompletableFuture<Trama>> futuros = new ArrayList<CompletableFuture<Trama>>();
        int[] inicios = new int[ns.size()];
        long[] t0 = new long[ns.size()];
        final long[] tFin = new long[ns.size()];
        int ini = 0;
        for (int i = 0; i < ns.size(); i++) {
            inicios[i] = ini;
            int c = cuotas[i];
            if (c == 0) {
                futuros.add(null);
                continue;
            }
            Escritor e = new Escritor(c * tam + c * 4 + 16).i32(version).i32(c).i32s(etq, ini, c).bytes(imgs, ini * tam, c * tam);
            t0[i] = System.nanoTime();
            final int idx = i;
            CompletableFuture<Trama> f = ns.get(i).enviar(Protocolo.TAREA_ENTRENAR, e.toBytes());
            f.whenComplete((t, ex) -> tFin[idx] = System.nanoTime());
            futuros.add(f);
            ini += c;
        }
        Paso total = new Paso();
        total.grad = new float[arq.totalParametros];
        for (int i = 0; i < ns.size(); i++) {
            CompletableFuture<Trama> f = futuros.get(i);
            if (f == null) {
                continue;
            }
            ConexionNodo n = ns.get(i);
            int c = cuotas[i];
            Paso parcial;
            try {
                Trama t = f.get(timeoutMs, TimeUnit.MILLISECONDS);
                double seg = ((tFin[i] > 0 ? tFin[i] : System.nanoTime()) - t0[i]) / 1e9;
                Lector l = t.lector();
                parcial = new Paso();
                parcial.n = l.i32();
                parcial.perdida = l.f32();
                parcial.aciertos = l.i32();
                l.i32();
                parcial.grad = l.f32s(l.i32());
                double v = c / Math.max(seg, 1e-6);
                n.velocidad = n.velocidadMedida ? 0.7 * n.velocidad + 0.3 * v : v;
                n.velocidadMedida = true;
            } catch (Exception ex) {
                System.out.println("\n[Entrenamiento] " + n + " fallo (" + ex.getMessage() + "); se recalcula localmente.");
                n.cerrar();
                parcial = pasoLocal(motorLocal, Arrays.copyOfRange(imgs, inicios[i] * tam, (inicios[i] + c) * tam),
                        Arrays.copyOfRange(etq, inicios[i], inicios[i] + c), c);
            }
            total.n += parcial.n;
            total.perdida += parcial.perdida;
            total.aciertos += parcial.aciertos;
            for (int j = 0; j < total.grad.length; j++) {
                total.grad[j] += parcial.grad[j];
            }
        }
        return total;
    }

    private void aplicarGradiente(float[] grad, int n, float lr) {
        float mu = cfg.real("entrenamiento.momentum", 0.9f);
        float wd = cfg.real("entrenamiento.weightDecay", 0.0001f);
        float[] p = pesos;
        float[] np = new float[p.length];
        float inv = 1f / Math.max(1, n);
        for (int i = 0; i < p.length; i++) {
            float g = grad[i] * inv + wd * p[i];
            momento[i] = mu * momento[i] - lr * g;
            np[i] = p[i] + momento[i];
        }
        pesos = np;
        version++;
    }

    private float tasaAprendizaje(String plan, float lr, float lrFinal, double avance) {
        if (plan.equals("coseno")) {
            return (float) (lrFinal + (lr - lrFinal) * 0.5 * (1 + Math.cos(Math.PI * avance)));
        }
        if (plan.equals("escalon")) {
            return avance < cfg.real("entrenamiento.lr.escalon", 0.75f) ? lr : lrFinal;
        }
        return lr;
    }

    private AumentoDatos crearAumento() {
        if (!Boolean.parseBoolean(cfg.txt("augmentation.activo", "false"))) {
            return null;
        }
        return new AumentoDatos(arq.c, arq.h, arq.w,
                cfg.real("augmentation.volteo.prob", 0.5f),
                cfg.ent("augmentation.desplazamiento", 4),
                AumentoDatos.Relleno.valueOf(cfg.txt("augmentation.relleno", "reflejo").toUpperCase(Locale.ROOT)),
                cfg.real("augmentation.rotacion.grados", 12f),
                cfg.real("augmentation.rotacion.prob", 0.5f),
                cfg.real("augmentation.color.prob", 0.8f),
                cfg.real("augmentation.brillo", 0.15f),
                cfg.real("augmentation.contraste", 0.2f),
                cfg.real("augmentation.ruido.prob", 0.25f),
                cfg.real("augmentation.ruido.sigma", 0.02f));
    }

    private Paso paso(MotorCNN motor, byte[] imgs, int[] etq, int n) throws Exception {
        return motor != null ? pasoLocal(motor, imgs, etq, n) : pasoDistribuido(imgs, etq, n);
    }

    public synchronized void entrenar(int epocas, boolean local, int maxMuestras) throws Exception {
        cargarDatos();
        if (train.n == 0) {
            System.out.println("[Entrenamiento] No hay imagenes de entrenamiento.");
            return;
        }
        entrenando = true;
        try {
            int lote = cfg.ent("entrenamiento.lote", 32);
            int n = maxMuestras > 0 ? Math.min(maxMuestras, train.n) : train.n;
            int tam = arq.tamImagen();
            Random rnd = new Random(1234 + epocasTotales);
            AumentoDatos aumento = crearAumento();
            Random rndAumento = new Random(cfg.ent("augmentation.semilla", 7) + 31L * epocasTotales);
            System.out.println("[Entrenamiento] Data augmentation: " + (aumento == null ? "desactivado" : aumento));
            float lr0 = cfg.real("entrenamiento.lr", 0.01f);
            float lrFinal = cfg.real("entrenamiento.lr.final", lr0 / 100);
            String plan = cfg.txt("entrenamiento.lr.plan", "constante").toLowerCase(Locale.ROOT);
            if (!plan.equals("coseno") && !plan.equals("escalon")) {
                plan = "constante";
            }
            System.out.printf(Locale.US, "[Entrenamiento] Learning rate: %s%n", plan.equals("constante")
                    ? String.format(Locale.US, "constante %.5f", lr0)
                    : String.format(Locale.US, "%s de %.5f a %.5f en %d epocas", plan, lr0, lrFinal, epocas));
            int pasosEpoca = (n + lote - 1) / lote;
            long pasosTotales = (long) epocas * pasosEpoca;
            float lr = lr0;
            int[] perm = new int[train.n];
            for (int i = 0; i < perm.length; i++) {
                perm[i] = i;
            }
            byte[] bImg = new byte[lote * tam];
            int[] bEtq = new int[lote];
            for (int ep = 1; ep <= epocas; ep++) {
                for (int i = perm.length - 1; i > 0; i--) {
                    int j = rnd.nextInt(i + 1);
                    int t = perm[i];
                    perm[i] = perm[j];
                    perm[j] = t;
                }
                long t0 = System.nanoTime();
                double perdida = 0;
                int aciertos = 0, vistos = 0, pasos = (n + lote - 1) / lote, siguienteAviso = 1;
                for (int s = 0; s < pasos; s++) {
                    int b = Math.min(lote, n - s * lote);
                    for (int i = 0; i < b; i++) {
                        int idx = perm[s * lote + i];
                        if (aumento != null) {
                            aumento.aplicar(train.imgs, idx * tam, bImg, i * tam, rndAumento);
                        } else {
                            System.arraycopy(train.imgs, idx * tam, bImg, i * tam, tam);
                        }
                        bEtq[i] = train.etiquetas[idx];
                    }
                    Paso p = paso(local ? motorLocal : null, bImg, bEtq, b);
                    lr = tasaAprendizaje(plan, lr0, lrFinal, ((long) (ep - 1) * pasosEpoca + s) / (double) pasosTotales);
                    aplicarGradiente(p.grad, p.n, lr);
                    perdida += p.perdida;
                    aciertos += p.aciertos;
                    vistos += p.n;
                    if ((s + 1) * 10 >= siguienteAviso * pasos) {
                        System.out.printf(Locale.US, "  epoca %d  %3d%%  perdida %.4f  precision %.2f%%  [%s]%n",
                                epocasTotales + ep, (s + 1) * 100 / pasos, perdida / vistos, 100.0 * aciertos / vistos,
                                local ? "local " + motorLocal.getHilos() + " hilos" : registro.vivos().size() + " nodos");
                        siguienteAviso = (s + 1) * 10 / pasos + 1;
                    }
                }
                double seg = (System.nanoTime() - t0) / 1e9;
                System.out.printf(Locale.US, "[Entrenamiento] Epoca %d lista: perdida %.4f | precision %.2f%% | lr %.5f | %.1f s | %.1f muestras/s%n",
                        epocasTotales + ep, perdida / vistos, 100.0 * aciertos / vistos, lr, seg, vistos / seg);
            }
            epocasTotales += epocas;
            guardarModelo();
        } finally {
            entrenando = false;
        }
    }

    private float[] inferirLote(final byte[] imgs, final int n, MotorCNN motor) throws Exception {
        if (motor != null) {
            return motor.inferir(arq, pesos, imgs, n);
        }
        final int k = arq.k, tam = arq.tamImagen();
        final float[] probs = new float[n * k];
        final ConcurrentLinkedQueue<int[]> cola = new ConcurrentLinkedQueue<int[]>();
        final java.util.concurrent.atomic.AtomicInteger cursor = new java.util.concurrent.atomic.AtomicInteger();
        final int bloque = cfg.ent("inferencia.bloque", 32);
        List<ConexionNodo> ns = registro.vivos();
        sincronizarTodos(ns);
        ns = registro.vivos();
        double vmax = 1e-9;
        for (ConexionNodo nodo : ns) {
            vmax = Math.max(vmax, nodo.velocidad);
        }
        List<Thread> hilos = new ArrayList<Thread>();
        for (final ConexionNodo nodo : ns) {
            final int miBloque = Math.max(1, (int) Math.round(bloque * nodo.velocidad / vmax));
            for (int h = 0; h < 2; h++) {
                Thread t = new Thread(() -> {
                    while (true) {
                        int[] b = cola.poll();
                        if (b == null) {
                            int ini = cursor.getAndAdd(miBloque);
                            if (ini >= n) {
                                return;
                            }
                            b = new int[]{ini, Math.min(miBloque, n - ini)};
                        }
                        try {
                            Escritor e = new Escritor(b[1] * tam + 8).i32(b[1]).bytes(imgs, b[0] * tam, b[1] * tam);
                            Trama r = nodo.solicitar(Protocolo.TAREA_INFERIR, e.toBytes(), timeoutMs);
                            Lector l = r.lector();
                            int cnt = l.i32();
                            l.i32();
                            l.i32();
                            l.f32s(probs, b[0] * k, cnt * k);
                        } catch (Exception ex) {
                            cola.add(b);
                            System.out.println("\n[Inferencia] " + nodo + " fallo: " + ex.getMessage());
                            nodo.cerrar();
                            return;
                        }
                    }
                });
                t.start();
                hilos.add(t);
            }
        }
        for (Thread t : hilos) {
            t.join();
        }
        int[] b;
        Arquitectura a = arq;
        float[] pw = pesos;
        int resto = cursor.get();
        if (resto < n) {
            cola.add(new int[]{resto, n - resto});
        }
        while ((b = cola.poll()) != null) {
            float[] p = motorLocal.inferir(a, pw, Arrays.copyOfRange(imgs, b[0] * tam, (b[0] + b[1]) * tam), b[1]);
            System.arraycopy(p, 0, probs, b[0] * k, p.length);
        }
        return probs;
    }

    /** Como inferirLote, pero promediando las probabilidades de las vistas TTA de cada imagen (por bloques). */
    private float[] inferirTta(byte[] imgs, int n, MotorCNN motor, int vistas) throws Exception {
        if (vistas == 1) {
            return inferirLote(imgs, n, motor);
        }
        int k = arq.k, tam = arq.tamImagen();
        float[] probs = new float[n * k];
        for (int ini = 0; ini < n; ini += 256) {
            int b = Math.min(256, n - ini);
            byte[] v = VistasTTA.generar(imgs, ini * tam, b, arq.c, arq.h, arq.w, vistas);
            float[] p = VistasTTA.promediar(inferirLote(v, b * vistas, motor), b, k, vistas);
            System.arraycopy(p, 0, probs, ini * k, p.length);
        }
        return probs;
    }

    public double evaluar(boolean local, int max) throws Exception {
        cargarDatos();
        Dataset d = test.n > 0 ? test : train;
        int n = max > 0 ? Math.min(max, d.n) : d.n;
        int vistas = vistasTta;
        long t0 = System.nanoTime();
        float[] probs = inferirTta(d.imgs, n, local ? motorLocal : null, vistas);
        double seg = (System.nanoTime() - t0) / 1e9;
        int k = arq.k, ok = 0;
        int[][] conf = new int[k][k];
        for (int i = 0; i < n; i++) {
            int pred = argmax(probs, i * k, k);
            conf[d.etiquetas[i]][pred]++;
            if (pred == d.etiquetas[i]) {
                ok++;
            }
        }
        double prec = 100.0 * ok / n;
        System.out.printf(Locale.US, "[Evaluacion] %d imagenes de %s | precision %.2f%% | %.2f s | %.1f img/s | %s | %s%n",
                n, d == test ? "prueba" : "entrenamiento", prec, seg, n / seg,
                local ? "local" : registro.vivos().size() + " nodos", vistas == 1 ? "sin TTA" : "TTA " + vistas + " vistas");
        StringBuilder sb = new StringBuilder("  Matriz de confusion (fila = real, columna = predicho)\n  ");
        sb.append(String.format("%12s", ""));
        for (int j = 0; j < k; j++) {
            sb.append(String.format("%10s", recortar(clases.get(j), 9)));
        }
        sb.append('\n');
        for (int i = 0; i < k; i++) {
            sb.append("  ").append(String.format("%12s", recortar(clases.get(i), 11)));
            for (int j = 0; j < k; j++) {
                sb.append(String.format("%10d", conf[i][j]));
            }
            sb.append('\n');
        }
        System.out.print(sb);
        return prec;
    }

    private static String recortar(String s, int n) {
        return s.length() <= n ? s : s.substring(0, n);
    }

    static int argmax(float[] a, int off, int n) {
        int m = 0;
        for (int i = 1; i < n; i++) {
            if (a[off + i] > a[off + m]) {
                m = i;
            }
        }
        return m;
    }

    public Prediccion predecir(byte[] img) throws Exception {
        if (pesos == null) {
            throw new IllegalStateException("Modelo no entrenado");
        }
        return "modelo".equals(modoInferencia) ? predecirModelo(img) : predecirDatos(img);
    }

    public Prediccion predecirDatos(byte[] img) throws Exception {
        List<ConexionNodo> excluidos = new ArrayList<ConexionNodo>();
        Prediccion pr = new Prediccion();
        int vistas = vistasTta;
        byte[] lote = vistas == 1 ? img : VistasTTA.generar(img, 0, 1, arq.c, arq.h, arq.w, vistas);
        for (int intento = 0; intento < 3; intento++) {
            ConexionNodo n = registro.menosCargado(excluidos);
            if (n == null) {
                break;
            }
            try {
                sincronizar(n);
                Trama t = n.solicitar(Protocolo.TAREA_INFERIR, new Escritor(lote.length + 8).i32(vistas).bytes(lote, 0, lote.length).toBytes(), timeoutMs);
                Lector l = t.lector();
                int cnt = l.i32();
                int k = l.i32();
                pr.msCalculo = l.i32();
                pr.probs = VistasTTA.promediar(l.f32s(cnt * k), 1, k, vistas);
                pr.trabajador = n.nombre + " (" + n.lenguaje.split(" ")[0] + ")";
                return pr;
            } catch (Exception ex) {
                excluidos.add(n);
            }
        }
        long t0 = System.nanoTime();
        pr.probs = VistasTTA.promediar(motorLocal.inferir(arq, pesos, lote, vistas), 1, arq.k, vistas);
        pr.msCalculo = (int) ((System.nanoTime() - t0) / 1000000L);
        pr.trabajador = "servidor (local)";
        return pr;
    }

    public Prediccion predecirModelo(byte[] img) throws Exception {
        long t0 = System.nanoTime();
        List<ConexionNodo> ns = registro.vivos();
        sincronizarTodos(ns);
        ns = registro.vivos();
        float[] x = new float[arq.tamImagen()];
        ModeloCNN.normalizar(img, 0, x);
        float[] a1 = capaDistribuida(Protocolo.CONV_FWD, 1, arq.f1, x, new int[]{arq.c, arq.h, arq.w}, (arq.h2 * arq.w2), ns);
        float[] a2 = capaDistribuida(Protocolo.CONV_FWD, 2, arq.f2, a1, new int[]{arq.f1, arq.h2, arq.w2}, (arq.h4 * arq.w4), ns);
        float[] a3 = capaDistribuida(Protocolo.MLP_FWD, 1, arq.oculta, a2, null, 1, ns);
        float[] o = capaDistribuida(Protocolo.MLP_FWD, 2, arq.k, a3, null, 1, ns);
        Prediccion pr = new Prediccion();
        pr.probs = new float[arq.k];
        CuboMLP.softmax(o, arq.k, pr.probs);
        pr.msCalculo = (int) ((System.nanoTime() - t0) / 1000000L);
        pr.trabajador = ns.isEmpty() ? "servidor (local, micro chunks)" : "micro chunks en " + ns.size() + " nodos";
        return pr;
    }

    private float[] capaDistribuida(int tipo, int capa, int total, float[] entrada, int[] dims, int porUnidad,
                                    List<ConexionNodo> ns) throws Exception {
        int partes = Math.max(1, Math.min(ns.size(), total));
        float[] salida = new float[total * porUnidad];
        List<CompletableFuture<Trama>> fs = new ArrayList<CompletableFuture<Trama>>();
        int[][] rangos = new int[partes][];
        for (int i = 0; i < partes; i++) {
            int ini = total * i / partes, fin = total * (i + 1) / partes;
            rangos[i] = new int[]{ini, fin};
            if (ns.isEmpty()) {
                fs.add(null);
                continue;
            }
            Escritor e = new Escritor(entrada.length * 4 + 32).i32(capa).i32(ini).i32(fin);
            if (tipo == Protocolo.CONV_FWD) {
                e.i32(dims[0]).i32(dims[1]).i32(dims[2]);
            } else {
                e.i32(entrada.length);
            }
            e.f32s(entrada);
            fs.add(ns.get(i).enviar(tipo, e.toBytes()));
        }
        for (int i = 0; i < partes; i++) {
            int ini = rangos[i][0], fin = rangos[i][1];
            float[] parte = null;
            if (fs.get(i) != null) {
                try {
                    Lector l = fs.get(i).get(timeoutMs, TimeUnit.MILLISECONDS).lector();
                    parte = l.f32s(l.i32());
                } catch (Exception ex) {
                    System.out.println("[Modelo] Micro chunk fallo en " + ns.get(i) + ": " + ex.getMessage());
                }
            }
            if (parte == null) {
                parte = tipo == Protocolo.CONV_FWD ? motorLocal.convForward(arq, pesos, capa, ini, fin, entrada)
                        : motorLocal.mlpForward(arq, pesos, capa, ini, fin, entrada);
            }
            System.arraycopy(parte, 0, salida, ini * porUnidad, parte.length);
        }
        return salida;
    }

    public void verificar() throws Exception {
        cargarDatos();
        Dataset d = test.n > 0 ? test : train;
        int n = Math.min(8, d.n);
        byte[] imgs = Arrays.copyOf(d.imgs, n * arq.tamImagen());
        float[] ref = motorLocal.inferir(arq, pesos, imgs, n);
        float[] x = new float[arq.tamImagen()];
        ModeloCNN.normalizar(imgs, 0, x);
        float[] refConv = motorLocal.convForward(arq, pesos, 1, 0, arq.f1, x);
        int[] etq = Arrays.copyOf(d.etiquetas, n);
        float[] refGrad = motorLocal.entrenar(arq, pesos, imgs, etq, n).gradiente;
        List<ConexionNodo> ns = registro.vivos();
        if (ns.isEmpty()) {
            System.out.println("[Verificar] No hay nodos conectados.");
            return;
        }
        System.out.println("[Verificar] Comparando " + n + " imagenes contra el calculo local en Java:");
        for (ConexionNodo nodo : ns) {
            try {
                sincronizar(nodo);
                Lector l = nodo.solicitar(Protocolo.TAREA_INFERIR, new Escritor().i32(n).bytes(imgs, 0, imgs.length).toBytes(), timeoutMs).lector();
                l.i32();
                l.i32();
                int ms = l.i32();
                float[] p = l.f32s(n * arq.k);
                Escritor e = new Escritor().i32(1).i32(0).i32(arq.f1).i32(arq.c).i32(arq.h).i32(arq.w).f32s(x);
                Lector lc = nodo.solicitar(Protocolo.CONV_FWD, e.toBytes(), timeoutMs).lector();
                float[] conv = lc.f32s(lc.i32());
                Escritor et = new Escritor().i32(version).i32(n).i32s(etq, 0, n).bytes(imgs, 0, imgs.length);
                Lector lg = nodo.solicitar(Protocolo.TAREA_ENTRENAR, et.toBytes(), timeoutMs).lector();
                lg.i32();
                lg.f32();
                lg.i32();
                lg.i32();
                float[] grad = lg.f32s(lg.i32());
                float escala = 0;
                for (float v : refGrad) {
                    escala = Math.max(escala, Math.abs(v));
                }
                float difGrad = maxDif(grad, refGrad) / Math.max(escala, 1e-12f);
                int igual = 0;
                for (int i = 0; i < n; i++) {
                    if (argmax(p, i * arq.k, arq.k) == argmax(ref, i * arq.k, arq.k)) {
                        igual++;
                    }
                }
                boolean ok = maxDif(p, ref) < 1e-3 && maxDif(conv, refConv) < 1e-3 && difGrad < 1e-3;
                System.out.printf(Locale.US, "  %-34s INFERIR dif %.1e (%d/%d igual, %d ms) | CONV_FWD dif %.1e | GRADIENTES dif.rel %.1e  %s%n",
                        nodo.toString(), maxDif(p, ref), igual, n, ms, maxDif(conv, refConv), difGrad, ok ? "OK" : "REVISAR");
            } catch (Exception ex) {
                System.out.println("  " + nodo + " ERROR: " + ex.getMessage());
            }
        }
    }

    private static float maxDif(float[] a, float[] b) {
        if (a.length != b.length) {
            return Float.POSITIVE_INFINITY;
        }
        float m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    public void benchmark(String tarea, int n, String destino, String etiqueta) throws Exception {
        cargarDatos();
        MotorCNN motor = null;
        int hilos;
        int nodos;
        String lenguajes;
        if (destino.startsWith("local")) {
            int h = destino.contains(":") ? Integer.parseInt(destino.substring(destino.indexOf(':') + 1)) : 1;
            motor = h == motorLocal.getHilos() ? motorLocal : new MotorCNN(h);
            hilos = h;
            nodos = 0;
            lenguajes = "Java(servidor)";
        } else {
            List<ConexionNodo> ns = registro.vivos();
            if (ns.isEmpty()) {
                System.out.println("[Bench] No hay nodos conectados para el destino 'cluster'.");
                return;
            }
            sincronizarTodos(ns);
            nodos = registro.vivos().size();
            hilos = registro.hilosTotales();
            lenguajes = registro.lenguajes();
        }
        if (etiqueta == null || etiqueta.isEmpty()) {
            etiqueta = motor != null ? (hilos == 1 ? "Secuencial" : "Paralelo " + hilos + "h")
                    : "Distribuido " + nodos + "n/" + hilos + "h";
        }
        long bytes0 = bytesRed();
        int tam = arq.tamImagen();
        double precision = Double.NaN;
        long t0 = System.nanoTime();
        if (tarea.equals("entrenar")) {
            n = Math.min(n, train.n);
            float[] copiaP = pesos, copiaM = momento.clone();
            int lote = cfg.ent("entrenamiento.lote", 32);
            int ok = 0;
            for (int s = 0; s < n; s += lote) {
                int b = Math.min(lote, n - s);
                Paso p = paso(motor, Arrays.copyOfRange(train.imgs, s * tam, (s + b) * tam),
                        Arrays.copyOfRange(train.etiquetas, s, s + b), b);
                aplicarGradiente(p.grad, p.n, cfg.real("entrenamiento.lr", 0.01f));
                ok += p.aciertos;
            }
            precision = 100.0 * ok / n;
            pesos = copiaP;
            momento = copiaM;
            version++;
        } else {
            Dataset d = test.n > 0 ? test : train;
            n = Math.min(n, d.n);
            float[] probs;
            if (tarea.equals("inferir")) {
                probs = inferirLote(d.imgs, n, motor);
            } else {
                probs = new float[n * arq.k];
                for (int i = 0; i < n; i++) {
                    byte[] img = Arrays.copyOfRange(d.imgs, i * tam, (i + 1) * tam);
                    Prediccion p;
                    if (motor != null) {
                        p = new Prediccion();
                        p.probs = motor.inferir(arq, pesos, img, 1);
                    } else {
                        p = tarea.equals("modelo") ? predecirModelo(img) : predecirDatos(img);
                    }
                    System.arraycopy(p.probs, 0, probs, i * arq.k, arq.k);
                }
            }
            int ok = 0;
            for (int i = 0; i < n; i++) {
                if (argmax(probs, i * arq.k, arq.k) == d.etiquetas[i]) {
                    ok++;
                }
            }
            precision = 100.0 * ok / n;
        }
        long ms = (System.nanoTime() - t0) / 1000000L;
        long red = bytesRed() - bytes0;
        if (motor != null && motor != motorLocal) {
            motor.close();
        }
        double porSeg = n / Math.max(ms / 1000.0, 1e-9);
        System.out.printf(Locale.US, "[Bench] %-22s %-8s n=%d nodos=%d hilos=%d -> %d ms | %.1f muestras/s | precision %.2f%% | red %.2f MB%n",
                etiqueta, tarea, n, nodos, hilos, ms, porSeg, precision, red / 1048576.0);
        File dir = new File(cfg.txt("resultados.dir", "resultados"));
        dir.mkdirs();
        File csv = new File(dir, "benchmarks.csv");
        boolean nuevo = !csv.exists();
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(csv, true), "UTF-8"));
        try {
            if (nuevo) {
                pw.println("fecha,etiqueta,tarea,destino,nodos,hilos_total,lenguajes,n,ms,muestras_s,precision,mb_red");
            }
            pw.printf(Locale.US, "%s,%s,%s,%s,%d,%d,%s,%d,%d,%.2f,%.2f,%.3f%n",
                    new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()), etiqueta.replace(',', ' '), tarea,
                    destino, nodos, hilos, lenguajes, n, ms, porSeg, precision, red / 1048576.0);
        } finally {
            pw.close();
        }
    }

    private long bytesRed() {
        long b = 0;
        for (ConexionNodo n : registro.vivos()) {
            b += n.bytesEnviados() + n.bytesRecibidos();
        }
        return b;
    }

    public String estado() {
        return String.format(Locale.US, "Modelo: %s | clases %s | epocas %d | version pesos %d | modo inferencia '%s' | TTA %d vista(s) | hilos locales %d%s",
                pesos == null ? "sin entrenar" : "listo", clases, epocasTotales, version, modoInferencia, vistasTta,
                motorLocal.getHilos(), entrenando ? " | ENTRENANDO" : "");
    }
}
