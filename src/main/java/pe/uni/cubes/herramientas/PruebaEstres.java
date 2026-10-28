package pe.uni.cubes.herramientas;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import pe.uni.cubes.cliente.ConexionCliente;
import pe.uni.cubes.cnn.ImagenUtil;

public final class PruebaEstres {

    private static final class Registro {
        int cliente, tx;
        long inicioMs, latenciaMs;
        boolean ok;
        String detalle;
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String host = "localhost", imagenes = "datos/imagenes/test", usuario = "estres", clave = "estres";
        int puerto = 6000, clientes = 20, tx = 50, dMin = 0, dMax = 200;
        for (int i = 0; i + 1 < args.length; i += 2) {
            String v = args[i + 1];
            switch (args[i]) {
                case "--servidor": host = v; break;
                case "--puerto": puerto = Integer.parseInt(v); break;
                case "--clientes": clientes = Integer.parseInt(v); break;
                case "--tx": tx = Integer.parseInt(v); break;
                case "--delayMin": dMin = Integer.parseInt(v); break;
                case "--delayMax": dMax = Integer.parseInt(v); break;
                case "--imagenes": imagenes = v; break;
                case "--usuario": usuario = v; break;
                case "--clave": clave = v; break;
                default: System.out.println("Opcion desconocida: " + args[i]);
            }
        }
        final List<byte[]> fotos = new ArrayList<byte[]>();
        final List<String> nombres = new ArrayList<String>();
        recolectar(new File(imagenes), fotos, nombres, 200);
        if (fotos.isEmpty()) {
            System.out.println("No se encontraron imagenes en " + imagenes);
            return;
        }
        System.out.printf("Prueba de estres: %d clientes x %d transacciones, delay aleatorio %d-%d ms, %d imagenes distintas%n",
                clientes, tx, dMin, dMax, fotos.size());

        final List<Registro> registros = Collections.synchronizedList(new ArrayList<Registro>());
        final AtomicInteger conectados = new AtomicInteger();
        final long inicio = System.currentTimeMillis();
        List<Thread> hilos = new ArrayList<Thread>();
        final String fHost = host, fUsuario = usuario, fClave = clave;
        final int fPuerto = puerto, fTx = tx, fMin = dMin, fMax = dMax;
        for (int c = 0; c < clientes; c++) {
            final int idc = c;
            Thread t = new Thread(() -> {
                Random r = new Random(idc * 31L + 7);
                ConexionCliente con;
                try {
                    con = ConexionCliente.conectar(fHost, fPuerto, fUsuario, fClave);
                    conectados.incrementAndGet();
                } catch (Exception ex) {
                    for (int i = 0; i < fTx; i++) {
                        Registro g = new Registro();
                        g.cliente = idc;
                        g.tx = i;
                        g.inicioMs = System.currentTimeMillis() - inicio;
                        g.detalle = "conexion: " + ex.getMessage();
                        registros.add(g);
                    }
                    return;
                }
                try {
                    for (int i = 0; i < fTx; i++) {
                        int k = r.nextInt(fotos.size());
                        Registro g = new Registro();
                        g.cliente = idc;
                        g.tx = i;
                        long t0 = System.nanoTime();
                        g.inicioMs = System.currentTimeMillis() - inicio;
                        try {
                            ConexionCliente.Respuesta resp = con.consultarSync("Identificar", nombres.get(k), fotos.get(k), 120000);
                            g.ok = resp.ok;
                            g.detalle = resp.ok ? resp.etiqueta + "@" + resp.trabajador : "error " + resp.codigoError;
                        } catch (Exception ex) {
                            g.detalle = "excepcion: " + ex.getMessage();
                        }
                        g.latenciaMs = (System.nanoTime() - t0) / 1000000L;
                        registros.add(g);
                        if (fMax > 0) {
                            Thread.sleep(fMin + r.nextInt(Math.max(1, fMax - fMin + 1)));
                        }
                    }
                } catch (InterruptedException ignored) {
                } finally {
                    con.close();
                }
            });
            t.start();
            hilos.add(t);
        }
        int total = clientes * tx;
        while (true) {
            boolean vivos = false;
            for (Thread t : hilos) {
                vivos |= t.isAlive();
            }
            System.out.printf("  progreso: %d/%d transacciones (%d clientes conectados)%n", registros.size(), total, conectados.get());
            if (!vivos) {
                break;
            }
            Thread.sleep(2000);
        }
        double dur = (System.currentTimeMillis() - inicio) / 1000.0;

        List<Long> lat = new ArrayList<Long>();
        int ok = 0, err503 = 0;
        for (Registro g : registros) {
            if (g.ok) {
                ok++;
                lat.add(g.latenciaMs);
            } else if (g.detalle != null && g.detalle.contains("503")) {
                err503++;
            }
        }
        Collections.sort(lat);
        double prom = 0;
        for (long l : lat) {
            prom += l;
        }
        prom = lat.isEmpty() ? 0 : prom / lat.size();
        long p50 = pct(lat, 50), p95 = pct(lat, 95), p99 = pct(lat, 99), max = lat.isEmpty() ? 0 : lat.get(lat.size() - 1);
        int err = registros.size() - ok;
        System.out.println("=============== RESULTADOS ===============");
        System.out.printf("Transacciones: %d | OK: %d | errores: %d (saturacion 503: %d)%n", registros.size(), ok, err, err503);
        System.out.printf("Duracion: %.2f s | Throughput: %.2f tx/s%n", dur, ok / dur);
        System.out.printf("Latencia (ms): promedio %.1f | p50 %d | p95 %d | p99 %d | max %d%n", prom, p50, p95, p99, max);

        File dir = new File("resultados");
        dir.mkdirs();
        PrintWriter d = new PrintWriter(new OutputStreamWriter(new FileOutputStream(new File(dir, "estres_detalle.csv")), "UTF-8"));
        d.println("cliente,tx,inicio_ms,latencia_ms,ok,detalle");
        synchronized (registros) {
            for (Registro g : registros) {
                d.printf("%d,%d,%d,%d,%d,%s%n", g.cliente, g.tx, g.inicioMs, g.latenciaMs, g.ok ? 1 : 0,
                        g.detalle == null ? "" : g.detalle.replace(',', ' '));
            }
        }
        d.close();
        File res = new File(dir, "estres_resumen.csv");
        boolean nuevo = !res.exists();
        PrintWriter w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(res, true), "UTF-8"));
        if (nuevo) {
            w.println("fecha,clientes,tx_por_cliente,delay_min,delay_max,total,ok,errores,saturados,duracion_s,tx_s,lat_prom,p50,p95,p99,max");
        }
        w.printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.1f,%d,%d,%d,%d%n",
                new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()), clientes, tx, dMin, dMax,
                registros.size(), ok, err, err503, dur, ok / dur, prom, p50, p95, p99, max);
        w.close();
        System.out.println("Guardado en resultados/estres_detalle.csv y resultados/estres_resumen.csv");
    }

    private static long pct(List<Long> ordenada, int p) {
        if (ordenada.isEmpty()) {
            return 0;
        }
        int i = (int) Math.ceil(p / 100.0 * ordenada.size()) - 1;
        return ordenada.get(Math.max(0, Math.min(i, ordenada.size() - 1)));
    }

    private static void recolectar(File f, List<byte[]> fotos, List<String> nombres, int max) throws Exception {
        if (fotos.size() >= max) {
            return;
        }
        if (f.isDirectory()) {
            File[] hijos = f.listFiles();
            if (hijos != null) {
                java.util.Arrays.sort(hijos);
                for (File h : hijos) {
                    recolectar(h, fotos, nombres, max);
                }
            }
        } else if (f.isFile() && ImagenUtil.esImagen(f.getName())) {
            fotos.add(Files.readAllBytes(f.toPath()));
            nombres.add(f.getName());
        }
    }
}
