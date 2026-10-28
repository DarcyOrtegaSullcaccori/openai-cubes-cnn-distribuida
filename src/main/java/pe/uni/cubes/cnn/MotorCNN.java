package pe.uni.cubes.cnn;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

public final class MotorCNN implements AutoCloseable {

    private static final class Estado {
        final Arquitectura a;
        final float[] p;
        final int version;
        final ModeloCNN modelo;

        Estado(Arquitectura a, float[] p, int version) {
            this.a = a;
            this.p = p;
            this.version = version;
            this.modelo = new ModeloCNN(a);
        }
    }

    public static final class ResultadoEntrenamiento {
        public int n;
        public float sumaPerdida;
        public int aciertos;
        public float[] gradiente;
        public long ms;
    }

    private interface Tarea<T> {
        T ejecutar(int ini, int fin) throws Exception;
    }

    private final int hilos;
    private final ExecutorService pool;
    private volatile Estado estado;
    private final ThreadLocal<Object[]> espacios = new ThreadLocal<Object[]>();

    public MotorCNN(int hilos) {
        this.hilos = Math.max(1, hilos);
        if (this.hilos > 1) {
            final AtomicInteger cont = new AtomicInteger();
            pool = Executors.newFixedThreadPool(this.hilos, new ThreadFactory() {
                @Override
                public Thread newThread(Runnable r) {
                    Thread t = new Thread(r, "cubo-hilo-" + cont.incrementAndGet());
                    t.setDaemon(true);
                    return t;
                }
            });
        } else {
            pool = null;
        }
    }

    public void cargarPesos(int version, Arquitectura a, float[] p) {
        if (p.length != a.totalParametros) {
            throw new IllegalArgumentException("Cantidad de parametros invalida: " + p.length + " != " + a.totalParametros);
        }
        estado = new Estado(a, p, version);
    }

    public boolean tienePesos() {
        return estado != null;
    }

    public int getVersion() {
        Estado s = estado;
        return s == null ? -1 : s.version;
    }

    public Arquitectura getArquitectura() {
        Estado s = estado;
        return s == null ? null : s.a;
    }

    public int getHilos() {
        return hilos;
    }

    private Estado estado() {
        Estado s = estado;
        if (s == null) {
            throw new IllegalStateException("El motor aun no tiene pesos (falta SYNC_PESOS)");
        }
        return s;
    }

    private ModeloCNN.Espacio espacio(Estado s) {
        Object[] par = espacios.get();
        if (par == null || !s.a.igual((Arquitectura) par[0])) {
            par = new Object[]{s.a, s.modelo.nuevoEspacio()};
            espacios.set(par);
        }
        return (ModeloCNN.Espacio) par[1];
    }

    private <T> List<T> paralelo(int total, final Tarea<T> tarea) throws Exception {
        List<T> res = new ArrayList<T>();
        int partes = Math.min(hilos, total);
        if (partes <= 1 || pool == null) {
            res.add(tarea.ejecutar(0, total));
            return res;
        }
        List<Future<T>> futuros = new ArrayList<Future<T>>();
        for (int i = 0; i < partes; i++) {
            final int ini = (int) ((long) total * i / partes);
            final int fin = (int) ((long) total * (i + 1) / partes);
            futuros.add(pool.submit(new Callable<T>() {
                @Override
                public T call() throws Exception {
                    return tarea.ejecutar(ini, fin);
                }
            }));
        }
        try {
            for (Future<T> f : futuros) {
                res.add(f.get());
            }
        } catch (ExecutionException ex) {
            Throwable c = ex.getCause();
            if (c instanceof Exception) {
                throw (Exception) c;
            }
            throw ex;
        }
        return res;
    }

    public ResultadoEntrenamiento entrenar(byte[] imgs, int[] etiquetas, int n) throws Exception {
        return entrenar(estado(), imgs, etiquetas, n);
    }

    public ResultadoEntrenamiento entrenar(Arquitectura a, float[] p, byte[] imgs, int[] etiquetas, int n) throws Exception {
        return entrenar(new Estado(a, p, -1), imgs, etiquetas, n);
    }

    private ResultadoEntrenamiento entrenar(final Estado s, final byte[] imgs, final int[] etiquetas, int n) throws Exception {
        final int tam = s.a.tamImagen();
        long t0 = System.nanoTime();
        List<ResultadoEntrenamiento> partes = paralelo(n, new Tarea<ResultadoEntrenamiento>() {
            @Override
            public ResultadoEntrenamiento ejecutar(int ini, int fin) {
                ResultadoEntrenamiento r = new ResultadoEntrenamiento();
                r.gradiente = new float[s.a.totalParametros];
                ModeloCNN.Espacio e = espacio(s);
                for (int i = ini; i < fin; i++) {
                    r.sumaPerdida += s.modelo.entrenarMuestra(s.p, imgs, i * tam, etiquetas[i], r.gradiente, e);
                    if (e.acierto) {
                        r.aciertos++;
                    }
                }
                r.n = fin - ini;
                return r;
            }
        });
        ResultadoEntrenamiento total = partes.get(0);
        for (int i = 1; i < partes.size(); i++) {
            ResultadoEntrenamiento r = partes.get(i);
            total.n += r.n;
            total.sumaPerdida += r.sumaPerdida;
            total.aciertos += r.aciertos;
            for (int j = 0; j < total.gradiente.length; j++) {
                total.gradiente[j] += r.gradiente[j];
            }
        }
        total.ms = (System.nanoTime() - t0) / 1000000L;
        return total;
    }

    public float[] inferir(byte[] imgs, int n) throws Exception {
        return inferir(estado(), imgs, n);
    }

    public float[] inferir(Arquitectura a, float[] p, byte[] imgs, int n) throws Exception {
        return inferir(new Estado(a, p, -1), imgs, n);
    }

    private float[] inferir(final Estado s, final byte[] imgs, int n) throws Exception {
        final int tam = s.a.tamImagen();
        final float[] probs = new float[n * s.a.k];
        paralelo(n, new Tarea<Void>() {
            @Override
            public Void ejecutar(int ini, int fin) {
                ModeloCNN.Espacio e = espacio(s);
                for (int i = ini; i < fin; i++) {
                    s.modelo.predecir(s.p, imgs, i * tam, probs, i * s.a.k, e);
                }
                return null;
            }
        });
        return probs;
    }

    public float[] convForward(int capa, int fIni, int fFin, float[] entrada) throws Exception {
        return convForward(estado(), capa, fIni, fFin, entrada);
    }

    public float[] convForward(Arquitectura a, float[] p, int capa, int fIni, int fFin, float[] entrada) throws Exception {
        return convForward(new Estado(a, p, -1), capa, fIni, fFin, entrada);
    }

    private float[] convForward(final Estado s, int capa, final int fIni, final int fFin, final float[] entrada) throws Exception {
        final Arquitectura a = s.a;
        final int C = capa == 1 ? a.c : a.f1;
        final int H = capa == 1 ? a.h : a.h2;
        final int W = capa == 1 ? a.w : a.w2;
        final int F = capa == 1 ? a.f1 : a.f2;
        final int wOff = capa == 1 ? a.oW1 : a.oW2;
        final int bOff = capa == 1 ? a.oB1 : a.oB2;
        if (entrada.length != C * H * W || fIni < 0 || fFin > F || fIni >= fFin) {
            throw new IllegalArgumentException("CONV_FWD con dimensiones invalidas");
        }
        final float[] z = new float[F * H * W];
        final float[] salida = new float[F * (H / 2) * (W / 2)];
        paralelo(fFin - fIni, new Tarea<Void>() {
            @Override
            public Void ejecutar(int ini, int fin) {
                CuboConvolucion.forward(entrada, C, H, W, s.p, wOff, bOff, fIni + ini, fIni + fin, z);
                CuboConvolucion.reluPool(z, fIni + ini, fIni + fin, H, W, salida, null);
                return null;
            }
        });
        int porFiltro = (H / 2) * (W / 2);
        float[] res = new float[(fFin - fIni) * porFiltro];
        System.arraycopy(salida, fIni * porFiltro, res, 0, res.length);
        return res;
    }

    public float[] mlpForward(int capa, int jIni, int jFin, float[] entrada) throws Exception {
        return mlpForward(estado(), capa, jIni, jFin, entrada);
    }

    public float[] mlpForward(Arquitectura a, float[] p, int capa, int jIni, int jFin, float[] entrada) throws Exception {
        return mlpForward(new Estado(a, p, -1), capa, jIni, jFin, entrada);
    }

    private float[] mlpForward(final Estado s, int capa, final int jIni, final int jFin, final float[] entrada) throws Exception {
        final Arquitectura a = s.a;
        final int nIn = capa == 1 ? a.plano : a.oculta;
        final int nOut = capa == 1 ? a.oculta : a.k;
        final int wOff = capa == 1 ? a.oW3 : a.oW4;
        final int bOff = capa == 1 ? a.oB3 : a.oB4;
        final boolean relu = capa == 1;
        if (entrada.length != nIn || jIni < 0 || jFin > nOut || jIni >= jFin) {
            throw new IllegalArgumentException("MLP_FWD con dimensiones invalidas");
        }
        final float[] salida = new float[nOut];
        paralelo(jFin - jIni, new Tarea<Void>() {
            @Override
            public Void ejecutar(int ini, int fin) {
                CuboMLP.forward(entrada, nIn, s.p, wOff, bOff, jIni + ini, jIni + fin, salida, relu);
                return null;
            }
        });
        float[] res = new float[jFin - jIni];
        System.arraycopy(salida, jIni, res, 0, res.length);
        return res;
    }

    @Override
    public void close() {
        if (pool != null) {
            pool.shutdownNow();
        }
    }
}
