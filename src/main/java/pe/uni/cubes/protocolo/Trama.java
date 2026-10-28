package pe.uni.cubes.protocolo;

public final class Trama {

    public final int tipo;
    public final int msgId;
    public final byte[] payload;

    public Trama(int tipo, int msgId, byte[] payload) {
        this.tipo = tipo;
        this.msgId = msgId;
        this.payload = payload;
    }

    public Lector lector() {
        return new Lector(payload);
    }

    @Override
    public String toString() {
        return Protocolo.nombre(tipo) + "#" + msgId + " (" + payload.length + " B)";
    }
}
