package pe.uni.cubes.servidor;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import pe.uni.cubes.protocolo.Conexion;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class ConexionNodo {

    public static final class ErrorRemoto extends Exception {
        public ErrorRemoto(String m) {
            super(m);
        }
    }

    private static final AtomicInteger SECUENCIA = new AtomicInteger(1);

    public final int id;
    public final String nombre;
    public final String lenguaje;
    public final int nucleos;
    public final int hilos;
    public final String cubos;
    private final Conexion con;
    private final RegistroNodos registro;
    private final Map<Integer, CompletableFuture<Trama>> pendientes = new ConcurrentHashMap<Integer, CompletableFuture<Trama>>();
    public final AtomicInteger enVuelo = new AtomicInteger();
    public final AtomicInteger tareasCompletadas = new AtomicInteger();
    public volatile boolean vivo = true;
    public volatile long ultimoContacto = System.currentTimeMillis();
    public volatile double velocidad;
    public volatile boolean velocidadMedida;
    public volatile int versionPesos = -1;

    ConexionNodo(int id, String nombre, String lenguaje, int nucleos, int hilos, String cubos, Conexion con,
                 RegistroNodos registro) {
        this.id = id;
        this.nombre = nombre;
        this.lenguaje = lenguaje;
        this.nucleos = nucleos;
        this.hilos = Math.max(1, hilos);
        this.cubos = cubos;
        this.con = con;
        this.registro = registro;
        this.velocidad = this.hilos;
    }

    void iniciarLector() {
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                bucleLectura();
            }
        }, "lector-nodo-" + id);
        t.setDaemon(true);
        t.start();
    }

    private void bucleLectura() {
        try {
            while (vivo) {
                Trama t = con.leer();
                ultimoContacto = System.currentTimeMillis();
                if (t.tipo == Protocolo.LATIDO) {
                    continue;
                }
                CompletableFuture<Trama> f = pendientes.remove(t.msgId);
                if (f == null) {
                    continue;
                }
                if (t.tipo == Protocolo.ERROR) {
                    Lector l = t.lector();
                    l.i32();
                    f.completeExceptionally(new ErrorRemoto("[" + nombre + "] " + l.str()));
                } else {
                    f.complete(t);
                }
            }
        } catch (IOException ex) {
            if (vivo) {
                System.out.println("\n[Nodos] Se perdio la conexion con " + this + " (" + ex.getMessage() + ")");
            }
        } finally {
            cerrar();
        }
    }

    public CompletableFuture<Trama> enviar(int tipo, byte[] payload) {
        final int msgId = SECUENCIA.incrementAndGet();
        CompletableFuture<Trama> f = new CompletableFuture<Trama>();
        if (!vivo) {
            f.completeExceptionally(new IOException("Nodo " + nombre + " desconectado"));
            return f;
        }
        pendientes.put(msgId, f);
        enVuelo.incrementAndGet();
        f.whenComplete((t, e) -> {
            enVuelo.decrementAndGet();
            if (e == null) {
                tareasCompletadas.incrementAndGet();
            }
        });
        try {
            con.enviar(tipo, msgId, payload);
        } catch (IOException ex) {
            pendientes.remove(msgId);
            f.completeExceptionally(ex);
            cerrar();
        }
        return f;
    }

    public Trama solicitar(int tipo, byte[] payload, long timeoutMs) throws Exception {
        try {
            return enviar(tipo, payload).get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (java.util.concurrent.ExecutionException ex) {
            Throwable c = ex.getCause();
            throw c instanceof Exception ? (Exception) c : ex;
        }
    }

    public void cerrar() {
        if (!vivo) {
            return;
        }
        vivo = false;
        con.close();
        for (CompletableFuture<Trama> f : pendientes.values()) {
            f.completeExceptionally(new IOException("Nodo " + nombre + " desconectado"));
        }
        pendientes.clear();
        registro.quitar(this);
    }

    public long bytesEnviados() {
        return con.getBytesEnviados();
    }

    public long bytesRecibidos() {
        return con.getBytesRecibidos();
    }

    public String remoto() {
        return con.remoto();
    }

    public double carga() {
        return enVuelo.get() / (double) hilos;
    }

    @Override
    public String toString() {
        return "#" + id + " " + nombre + " [" + lenguaje + ", " + hilos + " hilos]";
    }
}
