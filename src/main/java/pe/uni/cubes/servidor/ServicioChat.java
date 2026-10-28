package pe.uni.cubes.servidor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import pe.uni.cubes.cnn.ImagenUtil;
import pe.uni.cubes.protocolo.Conexion;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class ServicioChat {

    private final ModuloImagenes modulo;
    private final Map<String, String> usuarios = new HashMap<String, String>();
    private final Map<Integer, String> sesiones = new ConcurrentHashMap<Integer, String>();
    private final ThreadPoolExecutor pool;
    private final Random rnd = new Random();
    public final AtomicLong consultas = new AtomicLong();
    public final AtomicLong rechazadas = new AtomicLong();
    public final AtomicLong errores = new AtomicLong();
    public final AtomicLong msAcumulados = new AtomicLong();
    private volatile boolean verboso = true;

    public ServicioChat(Configuracion cfg, ModuloImagenes modulo) {
        this.modulo = modulo;
        int hilos = cfg.ent("chat.hilos", 8);
        pool = new ThreadPoolExecutor(hilos, hilos, 60, TimeUnit.SECONDS,
                new ArrayBlockingQueue<Runnable>(cfg.ent("chat.colaMax", 200)), new ThreadPoolExecutor.AbortPolicy());
        cargarUsuarios(cfg.txt("usuarios.archivo", "usuarios.txt"));
    }

    public void setVerboso(boolean v) {
        verboso = v;
    }

    private void cargarUsuarios(String archivo) {
        File f = new File(archivo);
        if (f.isFile()) {
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
                try {
                    String linea;
                    while ((linea = br.readLine()) != null) {
                        linea = linea.trim();
                        if (linea.isEmpty() || linea.startsWith("#") || !linea.contains(":")) {
                            continue;
                        }
                        int i = linea.indexOf(':');
                        usuarios.put(linea.substring(0, i).trim(), linea.substring(i + 1).trim());
                    }
                } finally {
                    br.close();
                }
            } catch (IOException ex) {
                System.out.println("[Chat] No se pudo leer " + archivo + ": " + ex.getMessage());
            }
        }
        if (usuarios.isEmpty()) {
            usuarios.put("admin", "admin");
        }
        System.out.println("[Chat] Usuarios habilitados: " + usuarios.keySet());
    }

    public void iniciar(int puerto) throws IOException {
        final ServerSocket ss = new ServerSocket(puerto);
        Thread t = new Thread(() -> {
            while (!ss.isClosed()) {
                try {
                    final Socket s = ss.accept();
                    Thread h = new Thread(() -> atenderCliente(s), "cliente-" + s.getPort());
                    h.setDaemon(true);
                    h.start();
                } catch (IOException ex) {
                    System.out.println("[Chat] Error aceptando: " + ex.getMessage());
                }
            }
        }, "aceptador-clientes");
        t.setDaemon(true);
        t.start();
        System.out.println("[Chat] Esperando clientes en el puerto " + puerto);
    }

    private void atenderCliente(Socket s) {
        Conexion c = null;
        Integer idChat = null;
        try {
            c = new Conexion(s);
            Trama login = c.leer();
            if (login.tipo != Protocolo.LOGIN) {
                return;
            }
            Lector l = login.lector();
            String usuario = l.str();
            String clave = l.str();
            String esperada = usuarios.get(usuario);
            if (esperada == null || !esperada.equals(clave)) {
                c.enviar(Protocolo.LOGIN_FALLO, login.msgId, new Escritor().str("Usuario o clave incorrectos").toBytes());
                return;
            }
            synchronized (sesiones) {
                do {
                    idChat = rnd.nextInt(10000);
                } while (sesiones.containsKey(idChat));
                sesiones.put(idChat, usuario);
            }
            StringBuilder cls = new StringBuilder();
            for (String k : modulo.getClases()) {
                cls.append(cls.length() > 0 ? "," : "").append(k);
            }
            c.enviar(Protocolo.LOGIN_OK, login.msgId, new Escritor().i32(idChat).str(usuario).str(cls.toString()).toBytes());
            if (verboso) {
                System.out.printf("[Chat] Usuario '%s' validado: idClientChat %04d (%s)%n", usuario, idChat, c.remoto());
            }
            while (true) {
                final Trama t = c.leer();
                if (t.tipo == Protocolo.LOGOUT) {
                    break;
                }
                if (t.tipo != Protocolo.CONSULTA) {
                    continue;
                }
                final Conexion cc = c;
                final long llegada = System.nanoTime();
                try {
                    pool.execute(() -> procesar(cc, t, llegada));
                } catch (RejectedExecutionException ex) {
                    rechazadas.incrementAndGet();
                    cc.enviar(Protocolo.ERROR, t.msgId, new Escritor().i32(Protocolo.ERR_SATURADO)
                            .str("Servidor saturado, intente nuevamente").toBytes());
                }
            }
        } catch (IOException ex) {
        } finally {
            if (idChat != null) {
                sesiones.remove(idChat);
            }
            if (c != null) {
                c.close();
            }
        }
    }

    private void procesar(Conexion c, Trama t, long llegada) {
        consultas.incrementAndGet();
        try {
            Lector l = t.lector();
            int idChat = l.i32();
            String texto = l.str();
            String archivo = l.str();
            byte[] img = l.blob();
            if (!sesiones.containsKey(idChat)) {
                throw new IllegalStateException("Sesion invalida");
            }
            if (img.length == 0) {
                c.enviar(Protocolo.RESPUESTA, t.msgId, new Escritor().str("Enviame una imagen con el texto 'Identificar'")
                        .f32(0f).str("").i32(0).str("servidor").toBytes());
                return;
            }
            if (!modulo.tieneModelo()) {
                c.enviar(Protocolo.ERROR, t.msgId, new Escritor().i32(Protocolo.ERR_SIN_MODELO)
                        .str("El modelo aun no esta entrenado").toBytes());
                return;
            }
            byte[] tensor;
            try {
                tensor = ImagenUtil.aTensor(img, 32, 32);
            } catch (IOException ex) {
                c.enviar(Protocolo.ERROR, t.msgId, new Escritor().i32(Protocolo.ERR_IMAGEN).str(ex.getMessage()).toBytes());
                return;
            }
            ModuloImagenes.Prediccion p = modulo.predecir(tensor);
            int k = p.probs.length;
            Integer[] orden = new Integer[k];
            for (int i = 0; i < k; i++) {
                orden[i] = i;
            }
            final float[] pr = p.probs;
            java.util.Arrays.sort(orden, (a, b) -> Float.compare(pr[b], pr[a]));
            StringBuilder top = new StringBuilder();
            for (int i = 0; i < Math.min(3, k); i++) {
                top.append(i > 0 ? ", " : "").append(modulo.getClases().get(orden[i]))
                        .append(String.format(Locale.US, " %.1f%%", 100 * pr[orden[i]]));
            }
            String etiqueta = modulo.getClases().get(orden[0]);
            int msTotal = (int) ((System.nanoTime() - llegada) / 1000000L);
            msAcumulados.addAndGet(msTotal);
            c.enviar(Protocolo.RESPUESTA, t.msgId, new Escritor().str(etiqueta).f32(pr[orden[0]]).str(top.toString())
                    .i32(msTotal).str(p.trabajador).toBytes());
            if (verboso) {
                System.out.printf(Locale.US, "[Chat] %04d \"%s\" %s -> %s (%.1f%%) en %d ms por %s%n", idChat, texto,
                        archivo, etiqueta, 100 * pr[orden[0]], msTotal, p.trabajador);
            }
        } catch (Exception ex) {
            errores.incrementAndGet();
            try {
                c.enviar(Protocolo.ERROR, t.msgId, new Escritor().i32(Protocolo.ERR_GENERAL).str(String.valueOf(ex.getMessage())).toBytes());
            } catch (IOException ignored) {
            }
        }
    }

    public String estado() {
        long n = consultas.get();
        return String.format(Locale.US, "Chat: %d sesiones | %d consultas | %d rechazadas (saturacion) | %d errores | %.1f ms promedio | cola %d/%d",
                sesiones.size(), n, rechazadas.get(), errores.get(), n == 0 ? 0.0 : msAcumulados.get() / (double) n,
                pool.getQueue().size(), pool.getQueue().size() + pool.getQueue().remainingCapacity());
    }
}
