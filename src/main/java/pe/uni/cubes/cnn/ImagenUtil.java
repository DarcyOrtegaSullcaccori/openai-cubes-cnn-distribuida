package pe.uni.cubes.cnn;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public final class ImagenUtil {

    private ImagenUtil() {
    }

    public static BufferedImage decodificar(byte[] datos) throws IOException {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(datos));
        if (img == null) {
            throw new IOException("Formato de imagen no soportado (use JPG, PNG, BMP o GIF)");
        }
        return img;
    }

    public static BufferedImage leer(File f) throws IOException {
        BufferedImage img = ImageIO.read(f);
        if (img == null) {
            throw new IOException("No se pudo leer la imagen: " + f);
        }
        return img;
    }

    public static BufferedImage redimensionar(BufferedImage src, int w, int h) {
        BufferedImage actual = aRGB(src);
        int cw = actual.getWidth(), ch = actual.getHeight();
        while (cw / 2 >= w && ch / 2 >= h) {
            cw /= 2;
            ch /= 2;
            actual = escalar(actual, cw, ch);
        }
        if (cw != w || ch != h) {
            actual = escalar(actual, w, h);
        }
        return actual;
    }

    private static BufferedImage aRGB(BufferedImage src) {
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, dst.getWidth(), dst.getHeight());
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return dst;
    }

    private static BufferedImage escalar(BufferedImage src, int w, int h) {
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(src, 0, 0, w, h, null);
        g.dispose();
        return dst;
    }

    public static byte[] aTensor(BufferedImage img, int w, int h) {
        BufferedImage r = redimensionar(img, w, h);
        byte[] t = new byte[3 * w * h];
        int plano = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = r.getRGB(x, y);
                int i = y * w + x;
                t[i] = (byte) ((rgb >> 16) & 0xFF);
                t[plano + i] = (byte) ((rgb >> 8) & 0xFF);
                t[2 * plano + i] = (byte) (rgb & 0xFF);
            }
        }
        return t;
    }

    public static byte[] aTensor(byte[] archivo, int w, int h) throws IOException {
        return aTensor(decodificar(archivo), w, h);
    }

    public static boolean esImagen(String nombre) {
        String n = nombre.toLowerCase();
        return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png") || n.endsWith(".bmp") || n.endsWith(".gif");
    }
}
