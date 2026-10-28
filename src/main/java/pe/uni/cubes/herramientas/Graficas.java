package pe.uni.cubes.herramientas;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;

public final class Graficas {

    private static final Color[] PALETA = {new Color(0x2F, 0x6B, 0xC4), new Color(0xE0, 0x7B, 0x26),
        new Color(0x2E, 0xA0, 0x43), new Color(0xC0, 0x39, 0x2B), new Color(0x7B, 0x4F, 0xB0),
        new Color(0x17, 0xA2, 0xB8), new Color(0x8C, 0x6D, 0x31), new Color(0x55, 0x55, 0x55)};

    static final class Serie {
        final String nombre;
        final List<double[]> puntos = new ArrayList<double[]>();

        Serie(String nombre) {
            this.nombre = nombre;
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String carpeta = "resultados";
        boolean soloPng = false;
        for (String a : args) {
            if (a.equals("--png")) {
                soloPng = true;
            } else {
                carpeta = a;
            }
        }
        File dir = new File(carpeta);
        File salida = new File(dir, "graficas");
        salida.mkdirs();
        Map<String, BufferedImage> imgs = new LinkedHashMap<String, BufferedImage>();

        List<Map<String, String>> bench = leerCsv(new File(dir, "benchmarks.csv"));
        Map<String, List<Map<String, String>>> porTarea = new LinkedHashMap<String, List<Map<String, String>>>();
        for (Map<String, String> f : bench) {
            String t = f.get("tarea");
            if (!porTarea.containsKey(t)) {
                porTarea.put(t, new ArrayList<Map<String, String>>());
            }
            porTarea.get(t).add(f);
        }
        for (Map.Entry<String, List<Map<String, String>>> e : porTarea.entrySet()) {
            Map<String, Map<String, String>> ult = new LinkedHashMap<String, Map<String, String>>();
            for (Map<String, String> f : e.getValue()) {
                ult.remove(f.get("etiqueta"));
                ult.put(f.get("etiqueta"), f);
            }
            List<String> cats = new ArrayList<String>(ult.keySet());
            double[] ms = new double[cats.size()];
            double base = -1;
            for (int i = 0; i < cats.size(); i++) {
                Map<String, String> f = ult.get(cats.get(i));
                ms[i] = num(f.get("ms"));
                if (base < 0 && "local:1".equals(f.get("destino"))) {
                    base = ms[i];
                }
            }
            if (base < 0) {
                base = ms.length > 0 ? ms[0] : 1;
            }
            String[] notas = new String[cats.size()];
            double[] speed = new double[cats.size()];
            for (int i = 0; i < cats.size(); i++) {
                speed[i] = base / Math.max(ms[i], 1e-9);
                notas[i] = String.format("%.0f ms  (x%.2f)", ms[i], speed[i]);
            }
            imgs.put("tiempos_" + e.getKey(), barras("Tiempo por escenario - tarea '" + e.getKey() + "'",
                    "Escenario", "Tiempo (ms)", cats, ms, notas));
            String[] notasS = new String[cats.size()];
            for (int i = 0; i < cats.size(); i++) {
                notasS[i] = String.format("x%.2f", speed[i]);
            }
            imgs.put("speedup_" + e.getKey(), barras("Speedup respecto al secuencial - tarea '" + e.getKey() + "'",
                    "Escenario", "Speedup (T_secuencial / T)", cats, speed, notasS));
            Map<String, Serie> series = new TreeMap<String, Serie>();
            for (Map<String, String> f : e.getValue()) {
                int nodos = (int) num(f.get("nodos"));
                if (nodos <= 0) {
                    continue;
                }
                int hpn = (int) Math.round(num(f.get("hilos_total")) / nodos);
                String nombre = hpn + " hilo(s) por nodo";
                if (!series.containsKey(nombre)) {
                    series.put(nombre, new Serie(nombre));
                }
                series.get(nombre).puntos.add(new double[]{nodos, num(f.get("ms"))});
            }
            if (!series.isEmpty()) {
                imgs.put("escalabilidad_" + e.getKey(), lineas("Escalabilidad - tarea '" + e.getKey() + "'",
                        "Número de nodos", "Tiempo (ms)", new ArrayList<Serie>(series.values()), true));
            }
        }

        List<Map<String, String>> res = leerCsv(new File(dir, "estres_resumen.csv"));
        if (!res.isEmpty()) {
            Serie tps = new Serie("throughput (tx/s)");
            Serie p95 = new Serie("latencia p95 (ms)");
            Serie prom = new Serie("latencia promedio (ms)");
            for (Map<String, String> f : res) {
                double c = num(f.get("clientes"));
                tps.puntos.add(new double[]{c, num(f.get("tx_s"))});
                p95.puntos.add(new double[]{c, num(f.get("p95"))});
                prom.puntos.add(new double[]{c, num(f.get("lat_prom"))});
            }
            List<Serie> l1 = new ArrayList<Serie>();
            l1.add(tps);
            imgs.put("estres_throughput", lineas("Prueba de estrés - throughput vs clientes concurrentes",
                    "Clientes concurrentes", "Transacciones / s", l1, true));
            List<Serie> l2 = new ArrayList<Serie>();
            l2.add(prom);
            l2.add(p95);
            imgs.put("estres_latencia", lineas("Prueba de estrés - latencia vs clientes concurrentes",
                    "Clientes concurrentes", "Latencia (ms)", l2, true));
        }
        List<Map<String, String>> det = leerCsv(new File(dir, "estres_detalle.csv"));
        if (!det.isEmpty()) {
            Serie ok = new Serie("respuestas OK");
            Serie err = new Serie("errores / saturación");
            for (Map<String, String> f : det) {
                double[] p = {num(f.get("inicio_ms")) / 1000.0, num(f.get("latencia_ms"))};
                ("1".equals(f.get("ok")) ? ok : err).puntos.add(p);
            }
            List<Serie> l = new ArrayList<Serie>();
            l.add(ok);
            if (!err.puntos.isEmpty()) {
                l.add(err);
            }
            imgs.put("estres_detalle", lineas("Última prueba de estrés - latencia de cada transacción",
                    "Instante de envío (s)", "Latencia (ms)", l, false));
        }

        if (imgs.isEmpty()) {
            System.out.println("No hay datos en " + dir.getAbsolutePath() + " (ejecute 'bench' en el servidor o PruebaEstres).");
            return;
        }
        for (Map.Entry<String, BufferedImage> e : imgs.entrySet()) {
            File f = new File(salida, e.getKey() + ".png");
            ImageIO.write(e.getValue(), "png", f);
            System.out.println("Grafica: " + f.getPath());
        }
        if (!soloPng && !GraphicsEnvironment.isHeadless()) {
            SwingUtilities.invokeLater(() -> {
                JFrame v = new JFrame("Open AI Cubes - Gráficas de desempeño");
                JTabbedPane tabs = new JTabbedPane();
                for (Map.Entry<String, BufferedImage> e : imgs.entrySet()) {
                    tabs.addTab(e.getKey(), new JScrollPane(new JLabel(new ImageIcon(e.getValue()))));
                }
                v.setContentPane(tabs);
                v.setSize(1000, 640);
                v.setLocationRelativeTo(null);
                v.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                v.setVisible(true);
            });
        }
    }

    static List<Map<String, String>> leerCsv(File f) throws IOException {
        List<Map<String, String>> filas = new ArrayList<Map<String, String>>();
        if (!f.isFile()) {
            return filas;
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
        try {
            String cab = br.readLine();
            if (cab == null) {
                return filas;
            }
            String[] cols = cab.split(",");
            String l;
            while ((l = br.readLine()) != null) {
                if (l.trim().isEmpty()) {
                    continue;
                }
                String[] v = l.split(",", -1);
                Map<String, String> m = new LinkedHashMap<String, String>();
                for (int i = 0; i < cols.length && i < v.length; i++) {
                    m.put(cols[i].trim(), v[i].trim());
                }
                filas.add(m);
            }
        } finally {
            br.close();
        }
        return filas;
    }

    static double num(String s) {
        try {
            return Double.parseDouble(s);
        } catch (Exception e) {
            return 0;
        }
    }

    private static final int ANCHO = 960, ALTO = 560, IZQ = 90, DER = 30, ARR = 60, ABA = 110;

    private static Graphics2D lienzo(BufferedImage img, String titulo) {
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, img.getWidth(), img.getHeight());
        g.setColor(new Color(0x22, 0x22, 0x22));
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString(titulo, IZQ, 34);
        return g;
    }

    private static double paso(double max) {
        double crudo = max / 6;
        double mag = Math.pow(10, Math.floor(Math.log10(Math.max(crudo, 1e-12))));
        double n = crudo / mag;
        return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
    }

    private static String fmt(double v) {
        if (Math.abs(v) >= 100 || v == Math.rint(v)) {
            return String.format("%.0f", v);
        }
        return String.format("%.2f", v);
    }

    private static double ejeY(Graphics2D g, double max, String etiqueta) {
        int alto = ALTO - ARR - ABA;
        double p = paso(max <= 0 ? 1 : max);
        double tope = Math.ceil((max <= 0 ? 1 : max) / p) * p;
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        FontMetrics fm = g.getFontMetrics();
        for (double v = 0; v <= tope + 1e-9; v += p) {
            int y = ALTO - ABA - (int) (v / tope * alto);
            g.setColor(new Color(0xE6, 0xE6, 0xE6));
            g.drawLine(IZQ, y, ANCHO - DER, y);
            g.setColor(new Color(0x55, 0x55, 0x55));
            String s = fmt(v);
            g.drawString(s, IZQ - 8 - fm.stringWidth(s), y + 4);
        }
        AffineTransform at = g.getTransform();
        g.rotate(-Math.PI / 2);
        g.setFont(new Font("SansSerif", Font.BOLD, 13));
        g.drawString(etiqueta, -(ARR + alto / 2 + g.getFontMetrics().stringWidth(etiqueta) / 2), 22);
        g.setTransform(at);
        g.setColor(new Color(0x88, 0x88, 0x88));
        g.drawLine(IZQ, ALTO - ABA, ANCHO - DER, ALTO - ABA);
        return tope;
    }

    static BufferedImage barras(String titulo, String ejeX, String ejeYt, List<String> cats, double[] v, String[] notas) {
        BufferedImage img = new BufferedImage(ANCHO, ALTO, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lienzo(img, titulo);
        double max = 0;
        for (double x : v) {
            max = Math.max(max, x);
        }
        double tope = ejeY(g, max * 1.12, ejeYt);
        int n = cats.size(), alto = ALTO - ARR - ABA, ancho = ANCHO - IZQ - DER;
        double hueco = ancho / (double) Math.max(n, 1);
        int bw = (int) Math.min(110, hueco * 0.6);
        for (int i = 0; i < n; i++) {
            int x = IZQ + (int) (hueco * i + (hueco - bw) / 2);
            int h = (int) (v[i] / tope * alto);
            g.setColor(PALETA[i % PALETA.length]);
            g.fillRect(x, ALTO - ABA - h, bw, h);
            g.setColor(new Color(0x22, 0x22, 0x22));
            g.setFont(new Font("SansSerif", Font.BOLD, 12));
            FontMetrics fm = g.getFontMetrics();
            String nota = notas[i];
            g.drawString(nota, x + bw / 2 - fm.stringWidth(nota) / 2, ALTO - ABA - h - 6);
            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            fm = g.getFontMetrics();
            List<String> lineas = partir(cats.get(i), fm, (int) hueco - 6);
            for (int j = 0; j < lineas.size() && j < 3; j++) {
                String s = lineas.get(j);
                g.drawString(s, x + bw / 2 - fm.stringWidth(s) / 2, ALTO - ABA + 18 + j * 15);
            }
        }
        g.setFont(new Font("SansSerif", Font.BOLD, 13));
        g.drawString(ejeX, IZQ + ancho / 2 - g.getFontMetrics().stringWidth(ejeX) / 2, ALTO - 20);
        g.dispose();
        return img;
    }

    private static List<String> partir(String s, FontMetrics fm, int ancho) {
        List<String> r = new ArrayList<String>();
        StringBuilder linea = new StringBuilder();
        for (String p : s.split(" ")) {
            if (linea.length() > 0 && fm.stringWidth(linea + " " + p) > ancho) {
                r.add(linea.toString());
                linea = new StringBuilder();
            }
            linea.append(linea.length() > 0 ? " " : "").append(p);
        }
        if (linea.length() > 0) {
            r.add(linea.toString());
        }
        return r;
    }

    static BufferedImage lineas(String titulo, String ejeX, String ejeYt, List<Serie> series, boolean unir) {
        BufferedImage img = new BufferedImage(ANCHO, ALTO, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = lienzo(img, titulo);
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE, maxY = 0;
        for (Serie s : series) {
            s.puntos.sort((a, b) -> Double.compare(a[0], b[0]));
            for (double[] p : s.puntos) {
                minX = Math.min(minX, p[0]);
                maxX = Math.max(maxX, p[0]);
                maxY = Math.max(maxY, p[1]);
            }
        }
        if (minX == maxX) {
            minX -= 1;
            maxX += 1;
        }
        if (unir && minX > 0) {
            minX = 0;
        }
        double tope = ejeY(g, maxY * 1.1, ejeYt);
        int alto = ALTO - ARR - ABA, ancho = ANCHO - IZQ - DER;
        double px = paso(maxX - minX);
        double x0 = Math.floor(minX / px) * px;
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        FontMetrics fm = g.getFontMetrics();
        for (double v = x0; v <= maxX + 1e-9; v += px) {
            if (v < minX - 1e-9) {
                continue;
            }
            int x = IZQ + (int) ((v - minX) / (maxX - minX) * ancho);
            g.setColor(new Color(0x55, 0x55, 0x55));
            g.drawLine(x, ALTO - ABA, x, ALTO - ABA + 4);
            String s = fmt(v);
            g.drawString(s, x - fm.stringWidth(s) / 2, ALTO - ABA + 18);
        }
        g.setFont(new Font("SansSerif", Font.BOLD, 13));
        g.drawString(ejeX, IZQ + ancho / 2 - g.getFontMetrics().stringWidth(ejeX) / 2, ALTO - 50);
        int leyY = ALTO - 26;
        int leyX = IZQ;
        for (int si = 0; si < series.size(); si++) {
            Serie s = series.get(si);
            Color c = PALETA[si % PALETA.length];
            g.setColor(c);
            g.setStroke(new BasicStroke(2.2f));
            int[] ant = null;
            for (double[] p : s.puntos) {
                int x = IZQ + (int) ((p[0] - minX) / (maxX - minX) * ancho);
                int y = ALTO - ABA - (int) (p[1] / tope * alto);
                if (unir && ant != null) {
                    g.drawLine(ant[0], ant[1], x, y);
                }
                int r = unir ? 4 : 2;
                g.fillOval(x - r, y - r, 2 * r, 2 * r);
                if (unir && s.puntos.size() <= 12) {
                    g.setFont(new Font("SansSerif", Font.PLAIN, 11));
                    g.drawString(fmt(p[1]), x + 6, y - 6);
                }
                ant = new int[]{x, y};
            }
            g.setStroke(new BasicStroke(1f));
            g.fillRect(leyX, leyY - 10, 14, 10);
            g.setColor(new Color(0x22, 0x22, 0x22));
            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            g.drawString(s.nombre, leyX + 20, leyY);
            leyX += 40 + g.getFontMetrics().stringWidth(s.nombre);
        }
        g.dispose();
        return img;
    }
}
