'use strict';

const os = require('os');

const MAGIC = 0x4F43;
const VERSION = 1;
const CABECERA = 12;
const MAX_PAYLOAD = 64 * 1024 * 1024;

const REGISTRO = 1, REGISTRO_OK = 2, LATIDO = 3, SYNC_PESOS = 4, ACK = 5;
const TAREA_ENTRENAR = 10, GRADIENTES = 11, TAREA_INFERIR = 12, PREDICCIONES = 13;
const CONV_FWD = 14, CONV_OUT = 15, MLP_FWD = 16, MLP_OUT = 17, ERROR = 20;
const ERR_GENERAL = 1;

const LITTLE = os.endianness() === 'LE';

function trama(tipo, msgId, payload) {
    const largo = payload ? payload.length : 0;
    const b = Buffer.allocUnsafe(CABECERA + largo);
    b.writeUInt16BE(MAGIC, 0);
    b.writeUInt8(VERSION, 2);
    b.writeUInt8(tipo, 3);
    b.writeInt32BE(msgId | 0, 4);
    b.writeInt32BE(largo, 8);
    if (largo) {
        payload.copy(b, CABECERA);
    }
    return b;
}

// Acumula los trozos que llegan por el socket y entrega tramas completas.
class Receptor {
    constructor(alRecibir) {
        this.alRecibir = alRecibir;
        this.trozos = [];
        this.disponible = 0;
        this.cab = null;
    }

    agregar(trozo) {
        this.trozos.push(trozo);
        this.disponible += trozo.length;
        for (;;) {
            if (this.cab === null) {
                if (this.disponible < CABECERA) {
                    return;
                }
                const c = this.tomar(CABECERA);
                if (c.readUInt16BE(0) !== MAGIC) {
                    throw new Error('MAGIC invalido');
                }
                if (c.readUInt8(2) !== VERSION) {
                    throw new Error('version no soportada: ' + c.readUInt8(2));
                }
                const largo = c.readInt32BE(8);
                if (largo < 0 || largo > MAX_PAYLOAD) {
                    throw new Error('longitud invalida: ' + largo);
                }
                this.cab = { tipo: c.readUInt8(3), msgId: c.readInt32BE(4), largo };
            }
            if (this.disponible < this.cab.largo) {
                return;
            }
            const t = this.cab;
            this.cab = null;
            this.alRecibir(t.tipo, t.msgId, this.tomar(t.largo));
        }
    }

    tomar(n) {
        if (n === 0) {
            return Buffer.alloc(0);
        }
        let b;
        if (this.trozos[0].length >= n) {
            b = this.trozos[0].subarray(0, n);
            this.trozos[0] = this.trozos[0].subarray(n);
            if (this.trozos[0].length === 0) {
                this.trozos.shift();
            }
        } else {
            b = Buffer.allocUnsafe(n);
            let pos = 0;
            while (pos < n) {
                const t = this.trozos[0];
                const k = Math.min(t.length, n - pos);
                t.copy(b, pos, 0, k);
                pos += k;
                if (k === t.length) {
                    this.trozos.shift();
                } else {
                    this.trozos[0] = t.subarray(k);
                }
            }
        }
        this.disponible -= n;
        return b;
    }
}

// Copia n floats big-endian a 'destino' (Float32Array en orden nativo).
function copiarF32(b, p, n, destino) {
    const vista = Buffer.from(destino.buffer, destino.byteOffset, n * 4);
    b.copy(vista, 0, p, p + n * 4);
    if (LITTLE) {
        vista.swap32();
    }
    return destino;
}

class Lector {
    constructor(b) {
        this.b = b;
        this.p = 0;
    }

    i32() {
        const v = this.b.readInt32BE(this.p);
        this.p += 4;
        return v;
    }

    f32() {
        const v = this.b.readFloatBE(this.p);
        this.p += 4;
        return v;
    }

    texto() {
        const n = this.i32();
        const s = this.b.toString('utf8', this.p, this.p + n);
        this.p += n;
        return s;
    }

    i32s(n) {
        const v = new Int32Array(n);
        for (let i = 0; i < n; i++) {
            v[i] = this.i32();
        }
        return v;
    }

    f32s(n, destino) {
        const v = copiarF32(this.b, this.p, n, destino || new Float32Array(n));
        this.p += 4 * n;
        return v;
    }

    bytes(n) {
        const v = this.b.subarray(this.p, this.p + n);
        this.p += n;
        return v;
    }
}

class Escritor {
    constructor() {
        this.partes = [];
    }

    i32(v) {
        const b = Buffer.allocUnsafe(4);
        b.writeInt32BE(v | 0, 0);
        this.partes.push(b);
        return this;
    }

    f32(v) {
        const b = Buffer.allocUnsafe(4);
        b.writeFloatBE(v, 0);
        this.partes.push(b);
        return this;
    }

    texto(s) {
        const b = Buffer.from(s, 'utf8');
        this.i32(b.length);
        this.partes.push(b);
        return this;
    }

    f32s(v) {
        const b = Buffer.from(new Uint8Array(v.buffer, v.byteOffset, v.length * 4));
        if (LITTLE) {
            b.swap32();
        }
        this.partes.push(b);
        return this;
    }

    aBytes() {
        return Buffer.concat(this.partes);
    }
}

module.exports = {
    MAGIC, VERSION, MAX_PAYLOAD,
    REGISTRO, REGISTRO_OK, LATIDO, SYNC_PESOS, ACK,
    TAREA_ENTRENAR, GRADIENTES, TAREA_INFERIR, PREDICCIONES,
    CONV_FWD, CONV_OUT, MLP_FWD, MLP_OUT, ERROR, ERR_GENERAL,
    trama, Receptor, Lector, Escritor, copiarF32,
};
