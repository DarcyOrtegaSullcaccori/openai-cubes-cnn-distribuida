package pe.uni.cubes.nodo;

import java.net.InetAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import pe.uni.cubes.cnn.Arquitectura;
import pe.uni.cubes.cnn.MotorCNN;
import pe.uni.cubes.protocolo.Conexion;
import pe.uni.cubes.protocolo.Escritor;
import pe.uni.cubes.protocolo.Lector;
import pe.uni.cubes.protocolo.Protocolo;
import pe.uni.cubes.protocolo.Trama;

public final class NodoJava {

    private final String host;
    private final int puerto;
    private final int hilos;
    private final String nombre;
    private MotorCNN motor;
    private Conexion con;

    public NodoJava(String host, int puerto, int hilos, String nombre) {
        this.host = host;
        this.puerto = puerto;
        this.hilos = hilos;
        this.nombre = nombre;
    }

    public static void main(String[] args) throws Exception {
        String host = "localhost";
        int puerto = 5000;
        int hilos = Runtime.getRuntime().availableProcessors();
        String nombre = "java-" + InetAddress.getLocalHost().getHostName();
        for (int i = 0; i + 1 < args.length; i += 2) {
            if (args[i].equals("--servidor")) {
                host = args[i + 1];
            } else if (args[i].equals("--puerto")) {
                puerto = Integer.parseInt(args[i + 1]);
            } else if (args[i].equals("--hilos")) {
                hilos = Integer.parseInt(args[i + 1]);
            } else if (args[i].equals("--nombre")) {
                nombre = args[i + 1];
            }
        }
        new NodoJava(host, puerto, hilos, nombre).ejecutar();
    }

    public void ejecutar() throws InterruptedException {
        motor = new MotorCNN(hilos);
        System.out.println("[NodoJava] " + nombre + " | hilos=" + hilos + " | nucleos="
                + Runtime.getRuntime().availableProcessors());
        while (true) {
            try {
                sesion();
            } catch (Exception ex) {
                System.out.println("[NodoJava] Conexion perdida/no disponible (" + (ex.getMessage() == null ? ex.getClass().getSimpleName() : ex.getMessage()) + "). Reintento en 3 s...");
            }
            if (con != null) {
                con.close();
            }
            Thread.sleep(3000);
        }
    }

    private void sesion() throws Exception {
        con = Conexion.conectar(host, puerto, 5000);
        final Conexion c = con;
        Escritor reg = new Escritor()
                .str(nombre).str("Java " + System.getProperty("java.version"))
                .i32(Runtime.getRuntime().availableProcessors()).i32(hilos).str("CONV,MLP");
        c.enviar(Protocolo.REGISTRO, 0, reg.toBytes());
        Trama ok = c.leer();
        if (ok.tipo != Protocolo.REGISTRO_OK) {
            throw new IllegalStateException("Registro rechazado: " + ok);
        }
        System.out.println("[NodoJava] Registrado en " + host + ":" + puerto + " con id " + ok.lector().i32());

        Thread latido = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    while (true) {
                        Thread.sleep(5000);
                        c.enviar(Protocolo.LATIDO, 0, null);
                    }
                } catch (Exception ignored) {
                }
            }
        }, "latido");
        latido.setDaemon(true);
        latido.start();

        ExecutorService manejadores = Executors.newFixedThreadPool(4);
        try {
            while (true) {
                final Trama t = c.leer();
                if (t.tipo == Protocolo.SYNC_PESOS) {
                    Lector l = t.lector();
                    int version = l.i32();
                    Arquitectura a = Arquitectura.leer(l);
                    int p = l.i32();
                    motor.cargarPesos(version, a, l.f32s(p));
                    c.enviar(Protocolo.ACK, t.msgId, new Escritor().i32(version).toBytes());
                    continue;
                }
                if (t.tipo == Protocolo.LATIDO) {
                    continue;
                }
                manejadores.submit(new Runnable() {
                    @Override
                    public void run() {
                        atender(c, t);
                    }
                });
            }
        } finally {
            manejadores.shutdownNow();
            latido.interrupt();
        }
    }

    private void atender(Conexion c, Trama t) {
        try {
            byte[] resp;
            int tipoResp;
            Lector l = t.lector();
            switch (t.tipo) {
                case Protocolo.TAREA_ENTRENAR: {
                    l.i32();
                    int n = l.i32();
                    int[] etq = l.i32s(n);
                    byte[] imgs = l.bytes(n * motor.getArquitectura().tamImagen());
                    MotorCNN.ResultadoEntrenamiento r = motor.entrenar(imgs, etq, n);
                    resp = new Escritor(r.gradiente.length * 4 + 64).i32(r.n).f32(r.sumaPerdida).i32(r.aciertos)
                            .i32((int) r.ms).i32(r.gradiente.length).f32s(r.gradiente).toBytes();
                    tipoResp = Protocolo.GRADIENTES;
                    break;
                }
                case Protocolo.TAREA_INFERIR: {
                    int n = l.i32();
                    byte[] imgs = l.bytes(n * motor.getArquitectura().tamImagen());
                    long t0 = System.nanoTime();
                    float[] probs = motor.inferir(imgs, n);
                    int ms = (int) ((System.nanoTime() - t0) / 1000000L);
                    resp = new Escritor(probs.length * 4 + 32).i32(n).i32(motor.getArquitectura().k).i32(ms)
                            .f32s(probs).toBytes();
                    tipoResp = Protocolo.PREDICCIONES;
                    break;
                }
                case Protocolo.CONV_FWD: {
                    int capa = l.i32(), fIni = l.i32(), fFin = l.i32();
                    int C = l.i32(), H = l.i32(), W = l.i32();
                    float[] out = motor.convForward(capa, fIni, fFin, l.f32s(C * H * W));
                    resp = new Escritor(out.length * 4 + 8).i32(out.length).f32s(out).toBytes();
                    tipoResp = Protocolo.CONV_OUT;
                    break;
                }
                case Protocolo.MLP_FWD: {
                    int capa = l.i32(), jIni = l.i32(), jFin = l.i32(), nIn = l.i32();
                    float[] out = motor.mlpForward(capa, jIni, jFin, l.f32s(nIn));
                    resp = new Escritor(out.length * 4 + 8).i32(out.length).f32s(out).toBytes();
                    tipoResp = Protocolo.MLP_OUT;
                    break;
                }
                default:
                    throw new IllegalArgumentException("Tipo no soportado: " + Protocolo.nombre(t.tipo));
            }
            c.enviar(tipoResp, t.msgId, resp);
        } catch (Exception ex) {
            try {
                c.enviar(Protocolo.ERROR, t.msgId,
                        new Escritor().i32(Protocolo.ERR_GENERAL).str(ex.getClass().getSimpleName() + ": " + ex.getMessage()).toBytes());
            } catch (Exception ignored) {
            }
        }
    }
}
