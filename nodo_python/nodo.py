#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import platform
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from operator import add

import modelo as M
import protocolo as P

_W = {}

def _iniciar_trabajador(arr, ver, arq):
    _W['arr'] = arr
    _W['ver'] = ver
    _W['arq'] = M.Arquitectura(*arq)
    _W['v'] = None
    _W['p'] = None

def _pesos():
    v = _W['ver'].value
    if v != _W['v']:
        _W['p'] = list(_W['arr'])
        _W['v'] = v
    return _W['p'], _W['arq']

def _t_entrenar(args):
    imgs, etq = args
    p, a = _pesos()
    return M.entrenar_lote(p, a, imgs, etq)

def _t_inferir(args):
    imgs, n = args
    p, a = _pesos()
    return M.inferir_lote(p, a, imgs, n)

def _t_conv(args):
    p, a = _pesos()
    return M.conv_micro(p, a, *args)

def _t_mlp(args):
    p, a = _pesos()
    return M.mlp_micro(p, a, *args)

def _partes(n, h):
    h = max(1, min(h, n))
    return [(n * i // h, n * (i + 1) // h) for i in range(h)]

class Motor:

    def __init__(self, hilos):
        self.hilos = max(1, hilos)
        self.pool = None
        self.arr = None
        self.ver = None
        self.arq = None
        self.p = None
        self.version = -1

    def cargar(self, version, arq, pesos):
        cambio = self.arq is None or self.arq.tupla() != arq.tupla()
        self.arq, self.p, self.version = arq, pesos, version
        if self.hilos == 1:
            return
        if self.pool is None or cambio:
            if self.pool is not None:
                self.pool.terminate()
            self.arr = mp.RawArray('f', len(pesos))
            self.ver = mp.RawValue('i', -1)
            self.arr[:] = pesos
            self.ver.value = version
            self.pool = mp.Pool(self.hilos, initializer=_iniciar_trabajador,
                                initargs=(self.arr, self.ver, arq.tupla()))
        else:
            self.arr[:] = pesos
            self.ver.value = version

    def entrenar(self, imgs, etq):
        a, n, tam = self.arq, len(etq), self.arq.tam_imagen()
        if self.hilos == 1:
            return M.entrenar_lote(self.p, a, imgs, etq)
        trozos = [(imgs[i * tam:j * tam], etq[i:j]) for i, j in _partes(n, self.hilos)]
        g, perdida, aciertos = None, 0.0, 0
        for gi, li, ai in self.pool.map(_t_entrenar, trozos):
            g = gi if g is None else list(map(add, g, gi))
            perdida += li
            aciertos += ai
        return g, perdida, aciertos

    def inferir(self, imgs, n):
        if self.hilos == 1:
            return M.inferir_lote(self.p, self.arq, imgs, n)
        tam = self.arq.tam_imagen()
        trozos = [(imgs[i * tam:j * tam], j - i) for i, j in _partes(n, self.hilos)]
        res = []
        for r in self.pool.map(_t_inferir, trozos):
            res.extend(r)
        return res

    def conv(self, capa, f_ini, f_fin, entrada):
        if self.hilos == 1:
            return M.conv_micro(self.p, self.arq, capa, f_ini, f_fin, entrada)
        trozos = [(capa, f_ini + i, f_ini + j, entrada) for i, j in _partes(f_fin - f_ini, self.hilos)]
        res = []
        for r in self.pool.map(_t_conv, trozos):
            res.extend(r)
        return res

    def mlp(self, capa, j_ini, j_fin, entrada):
        if self.hilos == 1:
            return M.mlp_micro(self.p, self.arq, capa, j_ini, j_fin, entrada)
        trozos = [(capa, j_ini + i, j_ini + j, entrada) for i, j in _partes(j_fin - j_ini, self.hilos)]
        res = []
        for r in self.pool.map(_t_mlp, trozos):
            res.extend(r)
        return res

def atender(sock, candado, motor, tipo, msg_id, payload):
    try:
        l = P.Lector(payload)
        if tipo == P.TAREA_ENTRENAR:
            l.i32()
            n = l.i32()
            etq = l.i32s(n)
            imgs = l.bytes(n * motor.arq.tam_imagen())
            t0 = time.perf_counter()
            g, perdida, aciertos = motor.entrenar(imgs, etq)
            ms = int((time.perf_counter() - t0) * 1000)
            resp = P.Escritor().i32(n).f32(perdida).i32(aciertos).i32(ms).i32(len(g)).f32s(g).a_bytes()
            tipo_r = P.GRADIENTES
        elif tipo == P.TAREA_INFERIR:
            n = l.i32()
            imgs = l.bytes(n * motor.arq.tam_imagen())
            t0 = time.perf_counter()
            probs = motor.inferir(imgs, n)
            ms = int((time.perf_counter() - t0) * 1000)
            resp = P.Escritor().i32(n).i32(motor.arq.k).i32(ms).f32s(probs).a_bytes()
            tipo_r = P.PREDICCIONES
        elif tipo == P.CONV_FWD:
            capa, f_ini, f_fin, C, H, W = [l.i32() for _ in range(6)]
            out = motor.conv(capa, f_ini, f_fin, l.f32s(C * H * W))
            resp = P.Escritor().i32(len(out)).f32s(out).a_bytes()
            tipo_r = P.CONV_OUT
        elif tipo == P.MLP_FWD:
            capa, j_ini, j_fin, n_in = [l.i32() for _ in range(4)]
            out = motor.mlp(capa, j_ini, j_fin, l.f32s(n_in))
            resp = P.Escritor().i32(len(out)).f32s(out).a_bytes()
            tipo_r = P.MLP_OUT
        else:
            raise ValueError('tipo no soportado: %d' % tipo)
        P.escribir_trama(sock, candado, tipo_r, msg_id, resp)
    except Exception as ex:
        try:
            P.escribir_trama(sock, candado, P.ERROR, msg_id,
                             P.Escritor().i32(P.ERR_GENERAL).texto('%s: %s' % (type(ex).__name__, ex)).a_bytes())
        except OSError:
            pass

def sesion(args, motor):
    sock = socket.create_connection((args.servidor, args.puerto), timeout=5)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    candado = threading.Lock()
    reg = (P.Escritor().texto(args.nombre).texto('Python ' + platform.python_version())
           .i32(os.cpu_count() or 1).i32(motor.hilos).texto('CONV,MLP').a_bytes())
    P.escribir_trama(sock, candado, P.REGISTRO, 0, reg)
    tipo, _, payload = P.leer_trama(sock)
    if tipo != P.REGISTRO_OK:
        raise ConnectionError('registro rechazado')
    print('[NodoPython] Registrado en %s:%d con id %d' % (args.servidor, args.puerto, P.Lector(payload).i32()), flush=True)

    activo = [True]

    def latidos():
        while activo[0]:
            time.sleep(5)
            try:
                P.escribir_trama(sock, candado, P.LATIDO, 0)
            except OSError:
                return

    threading.Thread(target=latidos, daemon=True).start()
    ejecutor = ThreadPoolExecutor(max_workers=4)
    try:
        while True:
            tipo, msg_id, payload = P.leer_trama(sock)
            if tipo == P.SYNC_PESOS:
                l = P.Lector(payload)
                version = l.i32()
                arq = M.Arquitectura(*[l.i32() for _ in range(7)])
                pesos = l.f32s(l.i32())
                motor.cargar(version, arq, pesos)
                P.escribir_trama(sock, candado, P.ACK, msg_id, P.Escritor().i32(version).a_bytes())
            elif tipo == P.LATIDO:
                continue
            else:
                ejecutor.submit(atender, sock, candado, motor, tipo, msg_id, payload)
    finally:
        activo[0] = False
        ejecutor.shutdown(wait=False)
        sock.close()

def main():
    ap = argparse.ArgumentParser(description='Nodo Python de Open AI Cubes')
    ap.add_argument('--servidor', default='localhost')
    ap.add_argument('--puerto', type=int, default=5000)
    ap.add_argument('--hilos', type=int, default=os.cpu_count() or 1, help='procesos de cálculo (H)')
    ap.add_argument('--nombre', default='python-' + socket.gethostname())
    args = ap.parse_args()
    motor = Motor(args.hilos)
    print('[NodoPython] %s | procesos=%d | núcleos=%d | Python %s' % (
        args.nombre, motor.hilos, os.cpu_count() or 1, platform.python_version()), flush=True)
    while True:
        try:
            sesion(args, motor)
        except (OSError, ConnectionError, ValueError) as ex:
            print('[NodoPython] Conexión perdida/no disponible (%s). Reintento en 3 s...' % ex, flush=True)
        time.sleep(3)

if __name__ == '__main__':
    mp.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
