import struct
import sys
from array import array

MAGIC = 0x4F43
VERSION = 1
MAX_PAYLOAD = 64 * 1024 * 1024
_CAB = struct.Struct('>HBBii')

REGISTRO, REGISTRO_OK, LATIDO, SYNC_PESOS, ACK = 1, 2, 3, 4, 5
TAREA_ENTRENAR, GRADIENTES, TAREA_INFERIR, PREDICCIONES = 10, 11, 12, 13
CONV_FWD, CONV_OUT, MLP_FWD, MLP_OUT, ERROR = 14, 15, 16, 17, 20
ERR_GENERAL = 1

_LITTLE = sys.byteorder == 'little'

def recibir_exacto(sock, n):
    buf = bytearray(n)
    vista = memoryview(buf)
    pos = 0
    while pos < n:
        r = sock.recv_into(vista[pos:], n - pos)
        if r == 0:
            raise ConnectionError('conexión cerrada por el servidor')
        pos += r
    return bytes(buf)

def leer_trama(sock):
    magic, version, tipo, msg_id, largo = _CAB.unpack(recibir_exacto(sock, 12))
    if magic != MAGIC:
        raise ValueError('MAGIC inválido: %x' % magic)
    if version != VERSION:
        raise ValueError('versión no soportada: %d' % version)
    if largo < 0 or largo > MAX_PAYLOAD:
        raise ValueError('longitud inválida: %d' % largo)
    return tipo, msg_id, recibir_exacto(sock, largo) if largo else b''

def escribir_trama(sock, candado, tipo, msg_id, payload=b''):
    datos = _CAB.pack(MAGIC, VERSION, tipo, msg_id, len(payload)) + payload
    with candado:
        sock.sendall(datos)

def floats_a_bytes(valores):
    a = array('f', valores)
    if _LITTLE:
        a.byteswap()
    return a.tobytes()

def bytes_a_floats(datos):
    a = array('f')
    a.frombytes(datos)
    if _LITTLE:
        a.byteswap()
    return a.tolist()

class Lector:
    def __init__(self, payload):
        self.b = payload
        self.p = 0

    def i32(self):
        v = struct.unpack_from('>i', self.b, self.p)[0]
        self.p += 4
        return v

    def f32(self):
        v = struct.unpack_from('>f', self.b, self.p)[0]
        self.p += 4
        return v

    def texto(self):
        n = self.i32()
        s = self.b[self.p:self.p + n].decode('utf-8')
        self.p += n
        return s

    def i32s(self, n):
        v = list(struct.unpack_from('>%di' % n, self.b, self.p))
        self.p += 4 * n
        return v

    def f32s(self, n):
        v = bytes_a_floats(self.b[self.p:self.p + 4 * n])
        self.p += 4 * n
        return v

    def bytes(self, n):
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

class Escritor:
    def __init__(self):
        self.partes = []

    def i32(self, v):
        self.partes.append(struct.pack('>i', v))
        return self

    def f32(self, v):
        self.partes.append(struct.pack('>f', v))
        return self

    def texto(self, s):
        b = s.encode('utf-8')
        self.i32(len(b))
        self.partes.append(b)
        return self

    def f32s(self, valores):
        self.partes.append(floats_a_bytes(valores))
        return self

    def a_bytes(self):
        return b''.join(self.partes)
