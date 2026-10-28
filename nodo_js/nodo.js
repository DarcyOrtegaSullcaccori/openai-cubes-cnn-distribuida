#!/usr/bin/env node
'use strict';

// Nodo JavaScript (Node.js >= 18) de Open AI Cubes. Solo biblioteca estandar:
// 'net' para el socket TCP y 'worker_threads' para los H hilos de calculo.
const net = require('net');
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');
const { performance } = require('perf_hooks');

const P = require('./protocolo');
const M = require('./modelo');

const fr = Math.fround;

function partes(total, h) {
    const n = Math.max(1, Math.min(h, total));
    const r = [];
    for (let i = 0; i < n; i++) {
        r.push([Math.floor(total * i / n), Math.floor(total * (i + 1) / n)]);
    }
    return r;
}

class Motor {
    constructor(hilos) {
        this.hilos = Math.max(1, hilos);
        this.estado = null;
        this.secuencia = 0;
        this.trabajadores = [];
        for (let i = 0; i < this.hilos; i++) {
            this.trabajadores.push(this.crearTrabajador());
        }
    }

    crearTrabajador() {
        const t = { w: new Worker(path.join(__dirname, 'trabajador.js')), pendientes: new Map() };
        t.w.on('message', (m) => {
            const p = t.pendientes.get(m.id);
            if (!p) {
                return;
            }
            t.pendientes.delete(m.id);
            if (m.error !== undefined) {
                p.reject(new Error(m.error));
            } else {
                p.resolve(m);
            }
        });
        const caido = (ex) => {
            for (const p of t.pendientes.values()) {
                p.reject(ex instanceof Error ? ex : new Error('hilo de calculo terminado (' + ex + ')'));
            }
            t.pendientes.clear();
            const i = this.trabajadores.indexOf(t);
            if (i >= 0) {
                this.trabajadores[i] = this.crearTrabajador();
            }
        };
        t.w.on('error', caido);
        t.w.on('exit', caido);
        return t;
    }

    cargar(version, arq, pesos) {
        this.estado = { version, arq, pesos };
    }

    actual() {
        if (this.estado === null) {
            throw new Error('el nodo aun no tiene pesos (falta SYNC_PESOS)');
        }
        return this.estado;
    }

    // Envia una parte al hilo con menos trabajo pendiente.
    ejecutar(s, msg, transferir) {
        let t = this.trabajadores[0];
        for (const o of this.trabajadores) {
            if (o.pendientes.size < t.pendientes.size) {
                t = o;
            }
        }
        const id = ++this.secuencia;
        msg.id = id;
        msg.arq = s.arq.tupla();
        msg.pesos = s.pesos.buffer;
        return new Promise((resolve, reject) => {
            t.pendientes.set(id, { resolve, reject });
            t.w.postMessage(msg, transferir || []);
        });
    }

    async entrenar(s, imgs, etq) {
        const tam = s.arq.tamImagen();
        const res = await Promise.all(partes(etq.length, this.hilos).map(([ini, fin]) => {
            const im = new Uint8Array(imgs.subarray(ini * tam, fin * tam));
            const et = etq.slice(ini, fin);
            return this.ejecutar(s, { op: 'entrenar', imgs: im, etq: et }, [im.buffer, et.buffer]);
        }));
        const g = res[0].g;
        let perdida = res[0].perdida, aciertos = res[0].aciertos;
        for (let k = 1; k < res.length; k++) {
            const gk = res[k].g;
            for (let j = 0; j < g.length; j++) {
                g[j] += gk[j];
            }
            perdida = fr(perdida + res[k].perdida);
            aciertos += res[k].aciertos;
        }
        return { g, perdida, aciertos };
    }

    async inferir(s, imgs, n) {
        const tam = s.arq.tamImagen(), k = s.arq.k;
        const probs = new Float32Array(n * k);
        await Promise.all(partes(n, this.hilos).map(async ([ini, fin]) => {
            const im = new Uint8Array(imgs.subarray(ini * tam, fin * tam));
            const m = await this.ejecutar(s, { op: 'inferir', imgs: im, n: fin - ini }, [im.buffer]);
            probs.set(m.r, ini * k);
        }));
        return probs;
    }

    async micro(s, op, capa, ini, fin, porUnidad, entrada) {
        const salida = new Float32Array((fin - ini) * porUnidad);
        await Promise.all(partes(fin - ini, this.hilos).map(async ([a, b]) => {
            const m = await this.ejecutar(s, { op, capa, ini: ini + a, fin: ini + b, entrada });
            salida.set(m.r, a * porUnidad);
        }));
        return salida;
    }
}

function leerPesos(l) {
    const version = l.i32();
    const arq = new M.Arquitectura(l.i32(), l.i32(), l.i32(), l.i32(), l.i32(), l.i32(), l.i32());
    const n = l.i32();
    if (n !== arq.total) {
        throw new Error('cantidad de parametros invalida: ' + n + ' != ' + arq.total);
    }
    const pesos = l.f32s(n, new Float32Array(new SharedArrayBuffer(n * 4)));
    return { version, arq, pesos };
}

// Lee la peticion y toma los pesos actuales de forma sincrona (antes de cualquier
// await), para que use la version vigente al momento de llegar, como en Go/Java.
function atender(motor, tipo, payload) {
    const l = new P.Lector(payload);
    const s = motor.actual();
    const a = s.arq;
    switch (tipo) {
        case P.TAREA_ENTRENAR: {
            l.i32();
            const n = l.i32();
            const etq = l.i32s(n);
            const imgs = l.bytes(n * a.tamImagen());
            const t0 = performance.now();
            return motor.entrenar(s, imgs, etq).then((r) => [P.GRADIENTES, new P.Escritor().i32(n).f32(r.perdida)
                .i32(r.aciertos).i32(Math.round(performance.now() - t0)).i32(r.g.length).f32s(r.g)]);
        }
        case P.TAREA_INFERIR: {
            const n = l.i32();
            const imgs = l.bytes(n * a.tamImagen());
            const t0 = performance.now();
            return motor.inferir(s, imgs, n).then((probs) => [P.PREDICCIONES, new P.Escritor().i32(n).i32(a.k)
                .i32(Math.round(performance.now() - t0)).f32s(probs)]);
        }
        case P.CONV_FWD: {
            const capa = l.i32(), ini = l.i32(), fin = l.i32();
            const C = l.i32(), H = l.i32(), W = l.i32();
            const [c, h, w, F] = M.dimsConv(a, capa);
            if ((capa !== 1 && capa !== 2) || C !== c || H !== h || W !== w || ini < 0 || fin > F || ini >= fin) {
                throw new Error('CONV_FWD con dimensiones invalidas');
            }
            return motor.micro(s, 'conv', capa, ini, fin, (H >> 1) * (W >> 1), l.f32s(C * H * W))
                .then((r) => [P.CONV_OUT, new P.Escritor().i32(r.length).f32s(r)]);
        }
        case P.MLP_FWD: {
            const capa = l.i32(), ini = l.i32(), fin = l.i32(), nIn = l.i32();
            const [ne, ns] = M.dimsMlp(a, capa);
            if ((capa !== 1 && capa !== 2) || nIn !== ne || ini < 0 || fin > ns || ini >= fin) {
                throw new Error('MLP_FWD con dimensiones invalidas');
            }
            return motor.micro(s, 'mlp', capa, ini, fin, 1, l.f32s(nIn))
                .then((r) => [P.MLP_OUT, new P.Escritor().i32(r.length).f32s(r)]);
        }
        default:
            throw new Error('tipo no soportado: ' + tipo);
    }
}

function sesion(opc, motor) {
    return new Promise((fin) => {
        const sock = net.connect({ host: opc.servidor, port: opc.puerto });
        let registrado = false, latido = null, terminado = false;
        const enviar = (tipo, msgId, payload) => {
            if (!sock.destroyed) {
                sock.write(P.trama(tipo, msgId, payload));
            }
        };
        const terminar = (motivo) => {
            if (terminado) {
                return;
            }
            terminado = true;
            clearInterval(latido);
            sock.destroy();
            fin(motivo);
        };
        const responderError = (msgId, ex) => enviar(P.ERROR, msgId,
            new P.Escritor().i32(P.ERR_GENERAL).texto((ex && ex.name || 'Error') + ': ' + (ex && ex.message || ex)).aBytes());

        const receptor = new P.Receptor((tipo, msgId, payload) => {
            if (!registrado) {
                if (tipo !== P.REGISTRO_OK) {
                    throw new Error('registro rechazado');
                }
                registrado = true;
                console.log(`[NodoJS] Registrado en ${opc.servidor}:${opc.puerto} con id ${new P.Lector(payload).i32()}`);
                latido = setInterval(() => enviar(P.LATIDO, 0, null), 5000);
                return;
            }
            if (tipo === P.SYNC_PESOS) {
                const e = leerPesos(new P.Lector(payload));
                motor.cargar(e.version, e.arq, e.pesos);
                enviar(P.ACK, msgId, new P.Escritor().i32(e.version).aBytes());
                return;
            }
            if (tipo === P.LATIDO) {
                return;
            }
            let promesa;
            try {
                promesa = atender(motor, tipo, payload);
            } catch (ex) {
                responderError(msgId, ex);
                return;
            }
            promesa.then(([tipoResp, esc]) => enviar(tipoResp, msgId, esc.aBytes()), (ex) => responderError(msgId, ex));
        });

        sock.setTimeout(5000, () => terminar('tiempo de conexion agotado'));
        sock.on('connect', () => {
            sock.setTimeout(0);
            sock.setNoDelay(true);
            const reg = new P.Escritor().texto(opc.nombre).texto('JavaScript ' + process.versions.node)
                .i32(nucleos()).i32(motor.hilos).texto('CONV,MLP').aBytes();
            enviar(P.REGISTRO, 0, reg);
        });
        sock.on('data', (trozo) => {
            try {
                receptor.agregar(trozo);
            } catch (ex) {
                terminar(ex.message);
            }
        });
        sock.on('error', (ex) => terminar(ex.code || ex.message));
        sock.on('close', () => terminar('conexion cerrada por el servidor'));
    });
}

function nucleos() {
    return typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
}

function argumentos(argv) {
    const opc = { servidor: 'localhost', puerto: 5000, hilos: nucleos(), nombre: 'js-' + os.hostname() };
    for (let i = 0; i < argv.length; i++) {
        let k = argv[i], v;
        if (!k.startsWith('--')) {
            continue;
        }
        k = k.slice(2);
        const eq = k.indexOf('=');
        if (eq >= 0) {
            v = k.slice(eq + 1);
            k = k.slice(0, eq);
        } else {
            v = argv[++i];
        }
        if (k === 'servidor' || k === 'nombre') {
            opc[k] = v;
        } else if (k === 'puerto' || k === 'hilos') {
            const n = parseInt(v, 10);
            if (!Number.isInteger(n) || n <= 0) {
                throw new Error(`valor invalido para --${k}: ${v}`);
            }
            opc[k] = n;
        } else {
            throw new Error(`opcion desconocida: --${k} (use --servidor, --puerto, --hilos, --nombre)`);
        }
    }
    return opc;
}

async function main() {
    let opc;
    try {
        opc = argumentos(process.argv.slice(2));
    } catch (ex) {
        console.error('[NodoJS] ' + ex.message);
        process.exit(2);
    }
    const motor = new Motor(opc.hilos);
    console.log(`[NodoJS] ${opc.nombre} | hilos=${motor.hilos} | nucleos=${nucleos()} | Node ${process.versions.node}`);
    for (;;) {
        const motivo = await sesion(opc, motor);
        console.log(`[NodoJS] Conexion perdida/no disponible (${motivo}). Reintento en 3 s...`);
        await new Promise((r) => setTimeout(r, 3000));
    }
}

process.on('SIGINT', () => process.exit(0));
main();
