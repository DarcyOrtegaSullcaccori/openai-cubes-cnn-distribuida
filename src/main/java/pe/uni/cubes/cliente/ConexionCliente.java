package pe.uni.cubes.cliente;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import pe.uni.cubes.protocolo.Conexion;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class ConexionCliente implements AutoCloseable {

    public static final class Respuesta {
        public boolean ok;
        public String etiqueta;
        public float probabilidad;
        public String top3;
        public int msServidor;
        public String trabajador;
        public int codigoError;
        public String error;
    }

    private final Conexion con;
    private final Map<Integer, CompletableFuture<Trama>> pendientes = new ConcurrentHashMap<Integer, CompletableFuture<Trama>>();
    private final AtomicInteger ids = new AtomicInteger();
    private volatile boolean abierta = true;
    public int idChat;
    public String usuario;
    public String clases;

    private ConexionCliente(Conexion con) {
        this.con = con;
    }

    public static ConexionCliente conectar(String host, int puerto, String usuario, String clave) throws IOException {
        Conexion c = Conexion.conectar(host, puerto, 5000);
        c.enviar(Protocolo.LOGIN, 1, new Escritor().str(usuario).str(clave).toBytes());
        c.setTimeoutLectura(10000);
        Trama t = c.leer();
        c.setTimeoutLectura(0);
        if (t.tipo != Protocolo.LOGIN_OK) {
            c.close();
            throw new IOException(t.tipo == Protocolo.LOGIN_FALLO ? t.lector().str() : "Respuesta inesperada");
        }
        ConexionCliente cc = new ConexionCliente(c);
        Lector l = t.lector();
        cc.idChat = l.i32();
        cc.usuario = l.str();
        cc.clases = l.str();
        cc.iniciarLector();
        return cc;
    }

    private void iniciarLector() {
        Thread t = new Thread(() -> {
            try {
                while (abierta) {
                    Trama r = con.leer();
                    CompletableFuture<Trama> f = pendientes.remove(r.msgId);
                    if (f != null) {
                        f.complete(r);
                    }
                }
            } catch (IOException ex) {
                for (CompletableFuture<Trama> f : pendientes.values()) {
                    f.completeExceptionally(ex);
                }
                pendientes.clear();
            }
        }, "lector-cliente");
        t.setDaemon(true);
        t.start();
    }

    public CompletableFuture<Respuesta> consultar(String texto, String archivo, byte[] imagen) {
        int id = ids.incrementAndGet();
        CompletableFuture<Trama> f = new CompletableFuture<Trama>();
        pendientes.put(id, f);
        try {
            con.enviar(Protocolo.CONSULTA, id, new Escritor(imagen.length + 64).i32(idChat).str(texto).str(archivo)
                    .blob(imagen).toBytes());
        } catch (IOException ex) {
            pendientes.remove(id);
            f.completeExceptionally(ex);
        }
        return f.thenApply(ConexionCliente::interpretar);
    }

    public Respuesta consultarSync(String texto, String archivo, byte[] imagen, long timeoutMs) throws Exception {
        try {
            return consultar(texto, archivo, imagen).get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (ExecutionException ex) {
            Throwable c = ex.getCause();
            throw c instanceof Exception ? (Exception) c : ex;
        }
    }

    private static Respuesta interpretar(Trama t) {
        Respuesta r = new Respuesta();
        Lector l = t.lector();
        if (t.tipo == Protocolo.RESPUESTA) {
            r.ok = true;
            r.etiqueta = l.str();
            r.probabilidad = l.f32();
            r.top3 = l.str();
            r.msServidor = l.i32();
            r.trabajador = l.str();
        } else {
            r.ok = false;
            r.codigoError = l.i32();
            r.error = l.str();
        }
        return r;
    }

    @Override
    public void close() {
        abierta = false;
        try {
            con.enviar(Protocolo.LOGOUT, 0, null);
        } catch (IOException ignored) {
        }
        con.close();
    }
}
