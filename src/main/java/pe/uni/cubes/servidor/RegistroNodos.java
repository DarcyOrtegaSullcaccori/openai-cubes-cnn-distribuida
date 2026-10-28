package pe.uni.cubes.servidor;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import pe.uni.cubes.protocolo.Conexion;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class RegistroNodos {

    private final List<ConexionNodo> nodos = new CopyOnWriteArrayList<ConexionNodo>();
    private final AtomicInteger ids = new AtomicInteger();
    private volatile Consumer<ConexionNodo> alRegistrar;
    private ServerSocket servidor;

    public void setAlRegistrar(Consumer<ConexionNodo> c) {
        alRegistrar = c;
    }

    public void iniciar(final int puerto) throws IOException {
        servidor = new ServerSocket(puerto);
        Thread aceptador = new Thread(new Runnable() {
            @Override
            public void run() {
                while (!servidor.isClosed()) {
                    try {
                        final Socket s = servidor.accept();
                        Thread t = new Thread(new Runnable() {
                            @Override
                            public void run() {
                                registrar(s);
                            }
                        }, "registro-nodo");
                        t.setDaemon(true);
                        t.start();
                    } catch (IOException ex) {
                        if (!servidor.isClosed()) {
                            System.out.println("[Nodos] Error aceptando: " + ex.getMessage());
                        }
                    }
                }
            }
        }, "aceptador-nodos");
        aceptador.setDaemon(true);
        aceptador.start();

        Thread vigia = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException e) {
                        return;
                    }
                    long ahora = System.currentTimeMillis();
                    for (ConexionNodo n : nodos) {
                        if (ahora - n.ultimoContacto > 30000 && n.enVuelo.get() == 0) {
                            System.out.println("\n[Nodos] " + n + " no envia latidos: se desconecta.");
                            n.cerrar();
                        }
                    }
                }
            }
        }, "vigia-latidos");
        vigia.setDaemon(true);
        vigia.start();
        System.out.println("[Nodos] Esperando nodos en el puerto " + puerto);
    }

    private void registrar(Socket s) {
        Conexion c = null;
        try {
            c = new Conexion(s);
            c.setTimeoutLectura(10000);
            Trama t = c.leer();
            if (t.tipo != Protocolo.REGISTRO) {
                throw new IOException("Se esperaba REGISTRO y llego " + Protocolo.nombre(t.tipo));
            }
            Lector l = t.lector();
            String nombre = l.str();
            String lenguaje = l.str();
            int nucleos = l.i32();
            int hilos = l.i32();
            String cubos = l.str();
            c.setTimeoutLectura(0);
            ConexionNodo n = new ConexionNodo(ids.incrementAndGet(), nombre, lenguaje, nucleos, hilos, cubos, c, this);
            c.enviar(Protocolo.REGISTRO_OK, t.msgId, new Escritor().i32(n.id).toBytes());
            nodos.add(n);
            n.iniciarLector();
            System.out.println("\n[Nodos] + Registrado " + n + " desde " + c.remoto() + " | cubos: " + cubos
                    + " | total nodos: " + nodos.size());
            Consumer<ConexionNodo> cb = alRegistrar;
            if (cb != null) {
                cb.accept(n);
            }
        } catch (IOException ex) {
            System.out.println("[Nodos] Registro fallido: " + ex.getMessage());
            if (c != null) {
                c.close();
            }
        }
    }

    void quitar(ConexionNodo n) {
        if (nodos.remove(n)) {
            System.out.println("[Nodos] - Retirado " + n + " | total nodos: " + nodos.size());
        }
    }

    public List<ConexionNodo> vivos() {
        List<ConexionNodo> r = new ArrayList<ConexionNodo>();
        for (ConexionNodo n : nodos) {
            if (n.vivo) {
                r.add(n);
            }
        }
        return r;
    }

    public ConexionNodo menosCargado(List<ConexionNodo> excluir) {
        ConexionNodo mejor = null;
        for (ConexionNodo n : nodos) {
            if (!n.vivo || (excluir != null && excluir.contains(n))) {
                continue;
            }
            if (mejor == null || n.carga() < mejor.carga()
                    || (n.carga() == mejor.carga() && n.velocidad > mejor.velocidad)) {
                mejor = n;
            }
        }
        return mejor;
    }

    public int hilosTotales() {
        int h = 0;
        for (ConexionNodo n : vivos()) {
            h += n.hilos;
        }
        return h;
    }

    public String lenguajes() {
        List<String> ls = new ArrayList<String>();
        for (ConexionNodo n : vivos()) {
            String l = n.lenguaje.split(" ")[0];
            if (!ls.contains(l)) {
                ls.add(l);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (String l : ls) {
            if (sb.length() > 0) {
                sb.append('+');
            }
            sb.append(l);
        }
        return sb.length() == 0 ? "-" : sb.toString();
    }
}
