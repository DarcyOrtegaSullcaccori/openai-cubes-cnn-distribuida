package pe.uni.cubes.protocolo;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public final class Protocolo {

    public static final int MAGIC = 0x4F43;
    public static final int VERSION = 1;
    public static final int CABECERA = 12;
    public static final int MAX_PAYLOAD = 64 * 1024 * 1024;

    public static final int REGISTRO = 1;
    public static final int REGISTRO_OK = 2;
    public static final int LATIDO = 3;
    public static final int SYNC_PESOS = 4;
    public static final int ACK = 5;
    public static final int TAREA_ENTRENAR = 10;
    public static final int GRADIENTES = 11;
    public static final int TAREA_INFERIR = 12;
    public static final int PREDICCIONES = 13;
    public static final int CONV_FWD = 14;
    public static final int CONV_OUT = 15;
    public static final int MLP_FWD = 16;
    public static final int MLP_OUT = 17;
    public static final int ERROR = 20;

    public static final int LOGIN = 30;
    public static final int LOGIN_OK = 31;
    public static final int LOGIN_FALLO = 32;
    public static final int CONSULTA = 33;
    public static final int RESPUESTA = 34;
    public static final int LOGOUT = 35;

    public static final int ERR_GENERAL = 1;
    public static final int ERR_SIN_MODELO = 2;
    public static final int ERR_IMAGEN = 3;
    public static final int ERR_SATURADO = 503;

    private Protocolo() {
    }

    public static String nombre(int tipo) {
        switch (tipo) {
            case REGISTRO: return "REGISTRO";
            case REGISTRO_OK: return "REGISTRO_OK";
            case LATIDO: return "LATIDO";
            case SYNC_PESOS: return "SYNC_PESOS";
            case ACK: return "ACK";
            case TAREA_ENTRENAR: return "TAREA_ENTRENAR";
            case GRADIENTES: return "GRADIENTES";
            case TAREA_INFERIR: return "TAREA_INFERIR";
            case PREDICCIONES: return "PREDICCIONES";
            case CONV_FWD: return "CONV_FWD";
            case CONV_OUT: return "CONV_OUT";
            case MLP_FWD: return "MLP_FWD";
            case MLP_OUT: return "MLP_OUT";
            case ERROR: return "ERROR";
            case LOGIN: return "LOGIN";
            case LOGIN_OK: return "LOGIN_OK";
            case LOGIN_FALLO: return "LOGIN_FALLO";
            case CONSULTA: return "CONSULTA";
            case RESPUESTA: return "RESPUESTA";
            case LOGOUT: return "LOGOUT";
            default: return "DESCONOCIDO(" + tipo + ")";
        }
    }

    public static void escribirTrama(DataOutputStream out, int tipo, int msgId, byte[] payload) throws IOException {
        int len = payload == null ? 0 : payload.length;
        out.writeShort(MAGIC);
        out.writeByte(VERSION);
        out.writeByte(tipo);
        out.writeInt(msgId);
        out.writeInt(len);
        if (len > 0) {
            out.write(payload);
        }
        out.flush();
    }

    public static Trama leerTrama(DataInputStream in) throws IOException {
        int magic = in.readUnsignedShort();
        if (magic != MAGIC) {
            throw new IOException("MAGIC invalido: 0x" + Integer.toHexString(magic));
        }
        int version = in.readUnsignedByte();
        if (version != VERSION) {
            throw new IOException("Version de protocolo no soportada: " + version);
        }
        int tipo = in.readUnsignedByte();
        int msgId = in.readInt();
        int len = in.readInt();
        if (len < 0 || len > MAX_PAYLOAD) {
            throw new IOException("Longitud de payload invalida: " + len);
        }
        byte[] payload = new byte[len];
        in.readFully(payload);
        return new Trama(tipo, msgId, payload);
    }
}
