# Protocolo OCP v1 (Open AI Cubes Protocol)

Protocolo binario propio sobre **TCP puro** (`java.net.Socket`, `socket` de Python, `net` de Go, `net` de Node.js).
No usa WebSocket, Socket.IO, HTTP, JSON, serialización de Java ni ningún middleware.

## 1. Formato de trama

Todos los enteros y flotantes viajan en **big-endian** (orden de red).

```
 0        2        3        4                 8                 12
 +--------+--------+--------+-----------------+-----------------+----------------------+
 | MAGIC  |VERSION |  TIPO  |     MSG_ID      |    LONGITUD     |  PAYLOAD (LONGITUD B) |
 | u16    |  u8    |  u8    |      i32        |      i32        |                      |
 | 0x4F43 |   1    |        |                 |                 |                      |
 +--------+--------+--------+-----------------+-----------------+----------------------+
```

- **MAGIC** `0x4F43` ("OC") detecta tramas corruptas o conexiones equivocadas.
- **MSG_ID** empareja cada respuesta con su petición. Gracias a él, un nodo puede tener **varias tareas en vuelo** sobre la misma conexión y responderlas en cualquier orden.
- **LONGITUD** máxima: 64 MB.

### Tipos primitivos del payload

| Tipo  | Codificación |
|-------|--------------|
| `i32` | entero con signo de 4 bytes |
| `f32` | flotante IEEE-754 de 4 bytes |
| `str` | `i32` con la longitud + bytes UTF-8 |
| `u8[n]` | n bytes crudos (imágenes ya preprocesadas, 0..255) |
| `blob` | `i32` con la longitud + bytes (archivo de imagen JPG/PNG) |

Equivalencias por lenguaje:

| Lenguaje | Implementación |
|----------|----------------|
| Java | `DataOutputStream` / `ByteBuffer` (big-endian por defecto) |
| Python | `struct.pack('>HBBii')` y `array('f')` + `byteswap()` |
| Go | `encoding/binary.BigEndian`, `math.Float32bits` |
| JavaScript (Node.js) | `Buffer` (`readInt32BE`, `writeFloatBE`…); los `f32[]` se copian a un `Float32Array` con `Buffer.swap32()` |

## 2. Mensajes Servidor ↔ Nodos (puerto 5000)

| Código | Mensaje | Dirección | Payload |
|---|---|---|---|
| 1 | REGISTRO | nodo → srv | `str nombre, str lenguaje, i32 núcleos, i32 hilos, str cubos ("CONV,MLP")` |
| 2 | REGISTRO_OK | srv → nodo | `i32 idNodo` |
| 3 | LATIDO | nodo → srv | (vacío), cada 5 s |
| 4 | SYNC_PESOS | srv → nodo | `i32 versión, 7×i32 arquitectura (C,H,W,F1,F2,OCULTA,K), i32 P, P×f32 pesos` |
| 5 | ACK | nodo → srv | `i32 versión` |
| 10 | TAREA_ENTRENAR | srv → nodo | `i32 versión, i32 n, n×i32 etiquetas, n×C·H·W u8 imágenes` |
| 11 | GRADIENTES | nodo → srv | `i32 n, f32 sumaPérdida, i32 aciertos, i32 msCálculo, i32 P, P×f32 suma de gradientes` |
| 12 | TAREA_INFERIR | srv → nodo | `i32 n, n×C·H·W u8 imágenes` |
| 13 | PREDICCIONES | nodo → srv | `i32 n, i32 K, i32 msCálculo, n·K×f32 probabilidades` |
| 14 | CONV_FWD (micro chunk) | srv → nodo | `i32 capa(1/2), i32 fIni, i32 fFin, i32 C, i32 H, i32 W, C·H·W×f32 entrada` |
| 15 | CONV_OUT | nodo → srv | `i32 cantidad, cantidad×f32` (conv + ReLU + MaxPool de los filtros [fIni,fFin)) |
| 16 | MLP_FWD (micro chunk) | srv → nodo | `i32 capa(1/2), i32 jIni, i32 jFin, i32 nEntrada, nEntrada×f32` |
| 17 | MLP_OUT | nodo → srv | `i32 cantidad, cantidad×f32` |
| 20 | ERROR | cualquiera | `i32 código, str mensaje` |

## 3. Mensajes Cliente ↔ Servidor (puerto 6000)

| Código | Mensaje | Dirección | Payload |
|---|---|---|---|
| 30 | LOGIN | cli → srv | `str usuario, str clave` |
| 31 | LOGIN_OK | srv → cli | `i32 idClientChat, str usuario, str clases (csv)` |
| 32 | LOGIN_FALLO | srv → cli | `str mensaje` |
| 33 | CONSULTA | cli → srv | `i32 idClientChat, str texto ("Identificar"), str nombreArchivo, blob imagen` |
| 34 | RESPUESTA | srv → cli | `str etiqueta, f32 probabilidad, str top3, i32 msTotal, str trabajador` |
| 35 | LOGOUT | cli → srv | (vacío) |
| 20 | ERROR | srv → cli | `i32 código, str mensaje`. Código 503 = servidor saturado, 2 = modelo sin entrenar, 3 = imagen inválida. |

## 4. Diagramas de secuencia

### 4.1 Registro de un nodo

```
Nodo (Java/Python/Go/JS)                      Servidor Central
   |------------- REGISTRO(nombre, lenguaje, núcleos, H) --->|
   |<------------ REGISTRO_OK(idNodo) -----------------------|
   |<------------ SYNC_PESOS(v, arquitectura, pesos) --------|  (si ya hay modelo)
   |------------- ACK(v) ----------------------------------->|
   |------------- LATIDO (cada 5 s) ------------------------>|  (sin latidos 30 s => se retira)
```

### 4.2 Un paso de entrenamiento distribuido (paralelismo de datos síncrono)

```
Servidor                          Nodo 1 (Java)     Nodo 2 (Go)      Nodo 3 (Python)
  | reparte el lote b entre nodos según su velocidad (minimiza el tiempo del paso)
  |-- SYNC_PESOS(v) ------------------>|                |                 |
  |-- SYNC_PESOS(v) ----------------------------------->|                 |
  |-- SYNC_PESOS(v) ------------------------------------------------------>|
  |<- ACK ... (de los tres) -----------|----------------|-----------------|
  |-- TAREA_ENTRENAR(n1 imgs) -------->|                |                 |
  |-- TAREA_ENTRENAR(n2 imgs) ------------------------->|                 |
  |-- TAREA_ENTRENAR(n3 imgs) ------------------------------------------->|
  |        (cada nodo reparte sus muestras entre sus H hilos/procesos)
  |<- GRADIENTES(Σg1) -----------------|                |                 |
  |<- GRADIENTES(Σg2) ----------------------------------|                 |
  |<- GRADIENTES(Σg3) ----------------------------------------------------|
  | g = (Σg1+Σg2+Σg3)/b ; SGD con momentum ; v = v+1
  |  (si un nodo falla o se desconecta, su parte se recalcula localmente en el servidor)
```

### 4.3 Consulta de chat, modo "datos" (balanceo de carga)

```
Cliente Swing              Servidor Central                         Nodo menos cargado
  |-- LOGIN --------------->|                                              |
  |<- LOGIN_OK(id 0812) ----|                                              |
  |-- CONSULTA(jpg) ------->| ImageIO: decodifica, 32x32, bytes CHW        |
  |                         |-- TAREA_INFERIR(1 imagen) ------------------>|
  |                         |<- PREDICCIONES(probs) -----------------------|
  |<- RESPUESTA("panda") ---|                                              |
```

### 4.4 Consulta de chat, modo "modelo" (micro chunks repartidos por capa)

```
Servidor                     Nodo 1            Nodo 2            Nodo 3
  |-- CONV_FWD(capa1, filtros 0-2) ->|                 |                 |
  |-- CONV_FWD(capa1, filtros 3-5) ------------------->|                 |
  |-- CONV_FWD(capa1, filtros 6-7) ------------------------------------->|
  |<- CONV_OUT x3 ; concatena => a1 (8x16x16)
  |-- CONV_FWD(capa2, ...) x3  => a2 (16x8x8 = 1024)
  |-- MLP_FWD(capa1, neuronas ...) x3 => a3 (64)
  |-- MLP_FWD(capa2, neuronas ...) x3 => logits (K) ; softmax en el servidor
```

## 5. Por qué así

- **Binario y no texto.** Una tarea de 32 imágenes ocupa unos 98 KB en binario. En texto ocuparía varias veces más y habría que parsearla.
- **Las imágenes viajan como `u8`, no como `f32`.** Ocupan 3 KB por imagen en lugar de 12 KB. Cada nodo normaliza con `x = byte/255 - 0.5`, igual en los cuatro lenguajes.
- **El servidor es el único que decodifica JPG/PNG.** Lo hace con `javax.imageio`. Python, Go y Node.js no tienen decodificadores en su biblioteca estándar, o no son equivalentes, así que reciben el tensor listo.
- **`SYNC_PESOS` se procesa en el hilo lector del nodo**, en orden. Así se garantiza que una `TAREA_ENTRENAR` posterior use los pesos nuevos.
- **`MSG_ID` y futuros en el servidor.** Permiten muchas consultas concurrentes por nodo, de modo que el chat escala con los nodos.
