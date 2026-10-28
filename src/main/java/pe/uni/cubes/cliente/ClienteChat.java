package pe.uni.cubes.cliente;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.datatransfer.DataFlavor;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.util.List;
import java.util.Locale;
import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.TransferHandler;
import javax.swing.UIManager;
import javax.swing.filechooser.FileNameExtensionFilter;

public final class ClienteChat extends JFrame {

    private static final Color AZUL = new Color(0x44, 0x72, 0xC4);
    private static final Color AZUL_OSCURO = new Color(0x2F, 0x52, 0x97);
    private static final Color VERDE = new Color(0x2E, 0xA0, 0x43);
    private static final Color GRIS = new Color(0x66, 0x66, 0x66);
    private static final Font BASE = new Font("SansSerif", Font.PLAIN, 14);

    private final ConexionCliente con;
    private final JPanel mensajes = new JPanel();
    private final JScrollPane scroll;
    private final JTextField texto = new JTextField("Identificar");
    private final JLabel miniatura = new JLabel();
    private File imagenSeleccionada;

    private ClienteChat(ConexionCliente con) {
        super("Open AI Cubes - Cliente");
        this.con = con;
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                con.close();
            }
        });

        JPanel raiz = new JPanel(new BorderLayout());
        raiz.setBackground(AZUL);
        raiz.setBorder(BorderFactory.createEmptyBorder(0, 10, 10, 10));

        JLabel titulo = new JLabel("Cliente");
        titulo.setForeground(Color.WHITE);
        titulo.setFont(new Font("SansSerif", Font.BOLD, 20));
        titulo.setBorder(BorderFactory.createEmptyBorder(10, 4, 10, 4));
        raiz.add(titulo, BorderLayout.NORTH);

        mensajes.setLayout(new BoxLayout(mensajes, BoxLayout.Y_AXIS));
        mensajes.setBackground(Color.WHITE);
        mensajes.setBorder(BorderFactory.createEmptyBorder(8, 10, 8, 10));
        JLabel validado = etiqueta("Usuario Validado", Font.BOLD, Color.BLACK);
        mensajes.add(validado);
        mensajes.add(etiqueta(String.format("idClientChat %04d", con.idChat), Font.PLAIN, Color.BLACK));
        mensajes.add(etiqueta("Usuario: " + con.usuario + "   |   clases: " + con.clases, Font.PLAIN, GRIS, 12));
        mensajes.add(Box.createVerticalStrut(8));
        JPanel envoltura = new JPanel(new BorderLayout());
        envoltura.setBackground(Color.WHITE);
        envoltura.add(mensajes, BorderLayout.NORTH);
        scroll = new JScrollPane(envoltura);
        scroll.setBorder(BorderFactory.createLineBorder(AZUL_OSCURO));
        scroll.getVerticalScrollBar().setUnitIncrement(16);
        scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        JPanel entrada = new JPanel(new BorderLayout(8, 0));
        entrada.setBackground(Color.WHITE);
        entrada.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(AZUL_OSCURO),
                BorderFactory.createEmptyBorder(6, 8, 6, 8)));
        texto.setFont(BASE);
        texto.setBorder(BorderFactory.createEmptyBorder());
        texto.addActionListener(e -> enviar());
        miniatura.setPreferredSize(new Dimension(56, 48));
        miniatura.setHorizontalAlignment(JLabel.CENTER);
        miniatura.setBorder(BorderFactory.createDashedBorder(GRIS, 3, 2));
        miniatura.setText("+img");
        miniatura.setForeground(GRIS);
        miniatura.setToolTipText("Clic para elegir una imagen (o arrástrela a la ventana)");
        miniatura.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        miniatura.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                elegirImagen();
            }
        });
        JPanel izq = new JPanel(new BorderLayout(8, 0));
        izq.setOpaque(false);
        izq.add(texto, BorderLayout.CENTER);
        izq.add(miniatura, BorderLayout.EAST);
        JButton enviar = new JButton(new IconoEnviar());
        enviar.setBorderPainted(false);
        enviar.setContentAreaFilled(false);
        enviar.setFocusPainted(false);
        enviar.setToolTipText("Enviar");
        enviar.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        enviar.addActionListener(e -> enviar());
        entrada.add(izq, BorderLayout.CENTER);
        entrada.add(enviar, BorderLayout.EAST);

        JPanel centro = new JPanel(new BorderLayout(0, 10));
        centro.setOpaque(false);
        centro.add(scroll, BorderLayout.CENTER);
        centro.add(entrada, BorderLayout.SOUTH);
        raiz.add(centro, BorderLayout.CENTER);
        setContentPane(raiz);

        TransferHandler th = new TransferHandler() {
            @Override
            public boolean canImport(TransferSupport s) {
                return s.isDataFlavorSupported(DataFlavor.javaFileListFlavor);
            }

            @Override
            @SuppressWarnings("unchecked")
            public boolean importData(TransferSupport s) {
                try {
                    List<File> fs = (List<File>) s.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                    if (!fs.isEmpty()) {
                        seleccionar(fs.get(0));
                    }
                    return true;
                } catch (Exception ex) {
                    return false;
                }
            }
        };
        raiz.setTransferHandler(th);
        texto.setTransferHandler(th);
        mensajes.setTransferHandler(th);

        setSize(470, 660);
        setLocationRelativeTo(null);
    }

    private static JLabel etiqueta(String s, int estilo, Color c) {
        return etiqueta(s, estilo, c, 14);
    }

    private static JLabel etiqueta(String s, int estilo, Color c, int tam) {
        JLabel l = new JLabel(s);
        l.setFont(new Font("SansSerif", estilo, tam));
        l.setForeground(c);
        l.setAlignmentX(Component.LEFT_ALIGNMENT);
        return l;
    }

    private void elegirImagen() {
        JFileChooser fc = new JFileChooser(imagenSeleccionada != null ? imagenSeleccionada.getParentFile() : new File("."));
        fc.setFileFilter(new FileNameExtensionFilter("Imágenes", "jpg", "jpeg", "png", "bmp", "gif"));
        if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            seleccionar(fc.getSelectedFile());
        }
    }

    private void seleccionar(File f) {
        try {
            BufferedImage img = ImageIO.read(f);
            if (img == null) {
                throw new IllegalArgumentException("formato no soportado");
            }
            imagenSeleccionada = f;
            miniatura.setText("");
            miniatura.setIcon(new ImageIcon(escalar(img, 52, 44)));
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "No se pudo abrir la imagen: " + ex.getMessage());
        }
    }

    private static Image escalar(BufferedImage img, int maxW, int maxH) {
        double r = Math.min(maxW / (double) img.getWidth(), maxH / (double) img.getHeight());
        return img.getScaledInstance(Math.max(1, (int) (img.getWidth() * r)), Math.max(1, (int) (img.getHeight() * r)),
                Image.SCALE_SMOOTH);
    }

    private void enviar() {
        if (imagenSeleccionada == null) {
            elegirImagen();
            if (imagenSeleccionada == null) {
                return;
            }
        }
        final File f = imagenSeleccionada;
        final String t = texto.getText().trim().isEmpty() ? "Identificar" : texto.getText().trim();
        byte[] datos;
        BufferedImage img;
        try {
            datos = Files.readAllBytes(f.toPath());
            img = ImageIO.read(f);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "No se pudo leer la imagen: " + ex.getMessage());
            return;
        }
        JPanel fila = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 2));
        fila.setOpaque(false);
        fila.setAlignmentX(Component.LEFT_ALIGNMENT);
        JLabel yo = etiqueta("Yo: " + t + "   ", Font.BOLD, AZUL_OSCURO);
        fila.add(yo);
        JLabel foto = new JLabel(new ImageIcon(escalar(img, 72, 60)));
        foto.setBorder(BorderFactory.createLineBorder(new Color(0xDD, 0xDD, 0xDD)));
        fila.add(foto);
        mensajes.add(fila);
        final JLabel bot = etiqueta("Bot: procesando...", Font.BOLD, GRIS);
        final JLabel detalle = etiqueta(" ", Font.PLAIN, GRIS, 11);
        mensajes.add(bot);
        mensajes.add(detalle);
        mensajes.add(Box.createVerticalStrut(10));
        refrescar();
        imagenSeleccionada = null;
        miniatura.setIcon(null);
        miniatura.setText("+img");
        final long t0 = System.nanoTime();
        con.consultar(t, f.getName(), datos).whenComplete((r, ex) -> SwingUtilities.invokeLater(() -> {
            long ms = (System.nanoTime() - t0) / 1000000L;
            if (ex != null) {
                bot.setText("Bot: (sin conexión con el servidor)");
                bot.setForeground(Color.RED);
            } else if (!r.ok) {
                bot.setText("Bot: " + r.error);
                bot.setForeground(new Color(0xC0, 0x39, 0x2B));
            } else {
                bot.setText("Bot: " + r.etiqueta);
                bot.setForeground(VERDE.darker());
                detalle.setText(String.format(Locale.US, "<html>%.1f%% &middot; %d ms &middot; resuelto por %s<br>top 3: %s</html>",
                        100 * r.probabilidad, ms, r.trabajador, r.top3));
            }
            refrescar();
        }));
    }

    private void refrescar() {
        mensajes.revalidate();
        mensajes.repaint();
        SwingUtilities.invokeLater(() -> {
            JScrollBar b = scroll.getVerticalScrollBar();
            b.setValue(b.getMaximum());
        });
    }

    private static final class IconoEnviar implements Icon {
        @Override
        public void paintIcon(Component c, Graphics g, int x, int y) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            Polygon p = new Polygon(new int[]{x + 4, x + 38, x + 4}, new int[]{y + 4, y + 18, y + 32}, 3);
            g2.setColor(VERDE);
            g2.fill(p);
            g2.setColor(VERDE.darker());
            g2.setStroke(new BasicStroke(1.5f));
            g2.draw(p);
            g2.dispose();
        }

        @Override
        public int getIconWidth() {
            return 42;
        }

        @Override
        public int getIconHeight() {
            return 36;
        }
    }

    public static void main(String[] args) throws Exception {
        String host = "localhost";
        String puerto = "6000";
        String usuario = "alumno";
        String clave = "";
        boolean auto = false;
        for (int i = 0; i + 1 < args.length; i += 2) {
            if (args[i].equals("--servidor")) {
                host = args[i + 1];
            } else if (args[i].equals("--puerto")) {
                puerto = args[i + 1];
            } else if (args[i].equals("--usuario")) {
                usuario = args[i + 1];
            } else if (args[i].equals("--clave")) {
                clave = args[i + 1];
                auto = true;
            }
        }
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
        }
        final String fHost = host, fPuerto = puerto, fUsuario = usuario, fClave = clave;
        final boolean fAuto = auto;
        SwingUtilities.invokeLater(() -> iniciarSesion(fHost, fPuerto, fUsuario, fClave, fAuto));
    }

    private static void iniciarSesion(String host, String puerto, String usuario, String clave, boolean auto) {
        JTextField tHost = new JTextField(host);
        JTextField tPuerto = new JTextField(puerto);
        JTextField tUsuario = new JTextField(usuario);
        JPasswordField tClave = new JPasswordField(clave);
        while (true) {
            if (!auto) {
                JPanel p = new JPanel(new GridLayout(0, 1, 4, 4));
                p.add(new JLabel("Servidor:"));
                p.add(tHost);
                p.add(new JLabel("Puerto:"));
                p.add(tPuerto);
                p.add(new JLabel("Usuario:"));
                p.add(tUsuario);
                p.add(new JLabel("Clave:"));
                p.add(tClave);
                int r = JOptionPane.showConfirmDialog(null, p, "Open AI Cubes - Iniciar sesión",
                        JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
                if (r != JOptionPane.OK_OPTION) {
                    System.exit(0);
                }
            }
            auto = false;
            try {
                ConexionCliente c = ConexionCliente.conectar(tHost.getText().trim(), Integer.parseInt(tPuerto.getText().trim()),
                        tUsuario.getText().trim(), new String(tClave.getPassword()));
                new ClienteChat(c).setVisible(true);
                return;
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(null, "No se pudo iniciar sesión: " + ex.getMessage(), "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        }
    }
}
