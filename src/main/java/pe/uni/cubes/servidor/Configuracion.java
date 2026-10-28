package pe.uni.cubes.servidor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.Properties;

public final class Configuracion {

    private final Properties p = new Properties();
    public final String archivo;

    public Configuracion(String archivo) {
        this.archivo = archivo;
        File f = new File(archivo);
        if (f.isFile()) {
            try {
                Reader r = new InputStreamReader(new FileInputStream(f), "UTF-8");
                try {
                    p.load(r);
                } finally {
                    r.close();
                }
            } catch (IOException ex) {
                System.out.println("[Config] No se pudo leer " + archivo + ": " + ex.getMessage());
            }
        } else {
            System.out.println("[Config] " + archivo + " no existe; se usan valores por defecto.");
        }
    }

    public String txt(String clave, String def) {
        String v = System.getProperty(clave, p.getProperty(clave));
        return v == null ? def : v.trim();
    }

    public int ent(String clave, int def) {
        return Integer.parseInt(txt(clave, String.valueOf(def)));
    }

    public float real(String clave, float def) {
        return Float.parseFloat(txt(clave, String.valueOf(def)));
    }

    public void poner(String clave, String valor) {
        p.setProperty(clave, valor);
    }
}
