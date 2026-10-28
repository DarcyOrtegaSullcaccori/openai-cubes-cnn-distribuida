package pe.uni.cubes.protocolo;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;

public final class Conexion implements Closeable {

    private final Socket socket;
    private final DataInputStream in;
    private final DataOutputStream out;
    private final Object candadoEnvio = new Object();
    private volatile long bytesEnviados;
    private volatile long bytesRecibidos;

    public Conexion(Socket socket) throws IOException {
        this.socket = socket;
        socket.setTcpNoDelay(true);
        socket.setKeepAlive(true);
        socket.setReceiveBufferSize(1 << 20);
        socket.setSendBufferSize(1 << 20);
        in = new DataInputStream(new BufferedInputStream(socket.getInputStream(), 1 << 16));
        out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream(), 1 << 16));
    }

    public static Conexion conectar(String host, int puerto, int timeoutMs) throws IOException {
        Socket s = new Socket();
        s.connect(new InetSocketAddress(host, puerto), timeoutMs);
        return new Conexion(s);
    }

    public void enviar(int tipo, int msgId, byte[] payload) throws IOException {
        synchronized (candadoEnvio) {
            Protocolo.escribirTrama(out, tipo, msgId, payload);
        }
        bytesEnviados += Protocolo.CABECERA + (payload == null ? 0 : payload.length);
    }

    public Trama leer() throws IOException {
        Trama t = Protocolo.leerTrama(in);
        bytesRecibidos += Protocolo.CABECERA + t.payload.length;
        return t;
    }

    public void setTimeoutLectura(int ms) throws IOException {
        socket.setSoTimeout(ms);
    }

    public String remoto() {
        return socket.getRemoteSocketAddress().toString();
    }

    public long getBytesEnviados() {
        return bytesEnviados;
    }

    public long getBytesRecibidos() {
        return bytesRecibidos;
    }

    @Override
    public void close() {
        try {
            socket.close();
        } catch (IOException ignored) {
        }
    }
}
