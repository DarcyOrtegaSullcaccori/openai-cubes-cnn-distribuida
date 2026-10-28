package pe.uni.cubes.herramientas;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.GradientPaint;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

public final class GenerarSinteticos {

    private static final String[] CLASES = {"circulo", "cuadrado", "triangulo", "cruz"};

    public static void main(String[] args) throws IOException {
        String carpeta = args.length > 0 ? args[0] : "datos/imagenes";
        int porClase = args.length > 1 ? Integer.parseInt(args[1]) : 300;
        Random r = new Random(7);
        int total = 0;
        for (int c = 0; c < CLASES.length; c++) {
            File dTrain = new File(carpeta, "train/" + CLASES[c]);
            File dTest = new File(carpeta, "test/" + CLASES[c]);
            dTrain.mkdirs();
            dTest.mkdirs();
            for (int i = 0; i < porClase; i++) {
                BufferedImage img = dibujar(c, r);
                File destino = new File(i % 7 == 0 ? dTest : dTrain, String.format("%s_%04d.png", CLASES[c], i));
                ImageIO.write(img, "png", destino);
                total++;
            }
        }
        System.out.println("Generadas " + total + " imagenes en " + new File(carpeta).getAbsolutePath());
    }

    static BufferedImage dibujar(int clase, Random r) {
        int t = 64;
        BufferedImage img = new BufferedImage(t, t, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        Color f1 = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256));
        Color f2 = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256));
        g.setPaint(new GradientPaint(0, 0, f1, t, t, f2));
        g.fillRect(0, 0, t, t);
        Color fig;
        do {
            fig = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256));
        } while (Math.abs(luz(fig) - (luz(f1) + luz(f2)) / 2) < 70);
        g.setColor(fig);
        int tam = 22 + r.nextInt(22);
        int cx = tam / 2 + 4 + r.nextInt(Math.max(1, t - tam - 8));
        int cy = tam / 2 + 4 + r.nextInt(Math.max(1, t - tam - 8));
        AffineTransform at = g.getTransform();
        g.rotate(r.nextDouble() * Math.PI / 2, cx, cy);
        switch (clase) {
            case 0:
                g.fillOval(cx - tam / 2, cy - tam / 2, tam, tam);
                break;
            case 1:
                g.fillRect(cx - tam / 2, cy - tam / 2, tam, tam);
                break;
            case 2:
                g.fillPolygon(new Polygon(new int[]{cx, cx - tam / 2, cx + tam / 2},
                        new int[]{cy - tam / 2, cy + tam / 2, cy + tam / 2}, 3));
                break;
            default:
                g.setStroke(new BasicStroke(tam / 4f));
                g.drawLine(cx - tam / 2, cy, cx + tam / 2, cy);
                g.drawLine(cx, cy - tam / 2, cx, cy + tam / 2);
        }
        g.setTransform(at);
        g.dispose();
        for (int i = 0; i < 300; i++) {
            int x = r.nextInt(t), y = r.nextInt(t);
            int v = r.nextInt(256);
            img.setRGB(x, y, new Color(v, v, v).getRGB());
        }
        return img;
    }

    private static int luz(Color c) {
        return (c.getRed() * 299 + c.getGreen() * 587 + c.getBlue() * 114) / 1000;
    }
}
