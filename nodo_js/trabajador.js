'use strict';

// Hilo de calculo (worker_thread). Recibe una parte de la tarea y los pesos
// en un SharedArrayBuffer (compartidos con el hilo principal, sin copiarlos).
const { parentPort } = require('worker_threads');
const M = require('./modelo');

let clave = null, arq = null, espacio = null;

function arquitectura(tupla) {
    const k = tupla.join(',');
    if (k !== clave) {
        clave = k;
        arq = new M.Arquitectura(...tupla);
        espacio = M.nuevoEspacio(arq);
    }
    return arq;
}

parentPort.on('message', (m) => {
    try {
        const a = arquitectura(m.arq);
        const p = new Float32Array(m.pesos);
        let r;
        switch (m.op) {
            case 'entrenar': {
                const t = M.entrenarLote(a, p, m.imgs, m.etq, espacio);
                parentPort.postMessage({ id: m.id, g: t.g, perdida: t.perdida, aciertos: t.aciertos }, [t.g.buffer]);
                return;
            }
            case 'inferir':
                r = M.inferirLote(a, p, m.imgs, m.n, espacio);
                break;
            case 'conv':
                r = M.convMicro(a, p, m.capa, m.ini, m.fin, m.entrada);
                break;
            case 'mlp':
                r = M.mlpMicro(a, p, m.capa, m.ini, m.fin, m.entrada);
                break;
            default:
                throw new Error('operacion desconocida: ' + m.op);
        }
        parentPort.postMessage({ id: m.id, r }, [r.buffer]);
    } catch (ex) {
        parentPort.postMessage({ id: m.id, error: String(ex && ex.message || ex) });
    }
});
