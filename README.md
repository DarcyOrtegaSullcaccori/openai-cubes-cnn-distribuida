# OpenAI Cubes — CNN Distribuida Políglota

Sistema de clasificación de imágenes basado en una CNN escrita desde cero, cuyo cálculo se reparte entre nodos en **Java, Python, Go y JavaScript (Node.js)** mediante un protocolo binario propio sobre TCP. Desarrollado para el curso CC4P1 (Computación Concurrente, UNI – 2026-II).

---

## Arquitectura

```
┌─────────────────────────── SERVIDOR CENTRAL (Java) ───────────────────────────┐
│  ServicioChat        ModuloImagenes (orquestador)        RegistroNodos         │
│  · auth/sesiones     · entrenamiento distribuido         · latidos             │
│  · pool acotado      · inferencia por lotes              · balanceo de carga   │
│  · cola + 503        · modo datos / modo modelo          · reconexión auto     │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │ TCP 5000 · OCP v1 (binario)
          ┌─────────────┬───────────────┴──────────────┬─────────────────┐
          ▼             ▼                               ▼                 ▼
    Nodo Java     Nodo Python               Nodo Go           Nodo Node.js
  ExecutorService multiprocessing         goroutines        worker_threads
  [Conv][MLP]     [Conv][MLP]             [Conv][MLP]       [Conv][MLP]

Cliente Swing ─── TCP 6000 ──► ServidorCentral
```

La CNN tiene ≈ 67 000 parámetros y sigue el esquema:

```
3×32×32 → Conv8+ReLU+Pool → 8×16×16
        → Conv16+ReLU+Pool → 16×8×8 = 1024
        → Dense64+ReLU → DenseK+Softmax
```

Entrenada con SGD + momentum + weight decay + learning rate coseno.

---

## Estructura

```
OpenAICubes/
├── pom.xml / nbactions.xml        Proyecto NetBeans (Maven, sin dependencias)
├── servidor.properties            Puertos, hiperparámetros, augmentation, TTA
├── usuarios.txt                   Usuarios del chat (usuario:clave)
├── cubes.sh / cubes.bat           Lanzadores sin Maven
├── src/main/java/pe/uni/cubes/
│   ├── protocolo/                 Protocolo, Trama, Escritor, Lector, Conexion
│   ├── cnn/                       CuboConvolucion, CuboMLP, ModeloCNN, MotorCNN,
│   │                              Dataset, AumentoDatos, VistasTTA, ArchivoModelo
│   ├── servidor/                  ServidorCentral, ModuloImagenes, ServicioChat,
│   │                              RegistroNodos, ConexionNodo, Configuracion
│   ├── nodo/                      NodoJava
│   ├── cliente/                   ClienteChat (Swing), ConexionCliente
│   └── herramientas/              PruebaEstres, Graficas, GenerarSinteticos
├── nodo_python/                   nodo.py, protocolo.py, cubos.py, modelo.py
├── nodo_go/                       main.go, protocolo.go, cubos.go, modelo.go
├── nodo_js/                       nodo.js, protocolo.js, cubos.js, modelo.js,
│                                  trabajador.js
├── modelos/modelo.ocw             Pesos entrenados (77.17 % / 78.17 % con TTA)
└── docs/PROTOCOLO.md              Especificación del protocolo OCP v1
```

**Sin dependencias externas.** Solo la biblioteca estándar de cada lenguaje. Maven únicamente descarga sus propios plugins de compilación.

---

## Requisitos

| Herramienta | Versión mínima |
|---|---|
| JDK | 8 |
| NetBeans | 12 |
| Python | 3.8 |
| Go | 1.20 |
| Node.js | 18 |

---

## Inicio rápido (una sola PC)

### 1. Compilar

Abra el proyecto en NetBeans (**File → Open Project** → carpeta `OpenAICubes`) y pulse **Build**. Sin NetBeans:

```bash
./cubes.sh compilar      # Linux/macOS
cubes.bat compilar       # Windows
```

### 2. Arrancar el servidor

```bash
./cubes.sh servidor
# o Run Project (F6) en NetBeans
```

### 3. Levantar los nodos

```bash
./cubes.sh nodo-java   --hilos 2 --nombre java-1
./cubes.sh nodo-python --hilos 2 --nombre py-1
./cubes.sh nodo-go     --hilos 2 --nombre go-1
./cubes.sh nodo-js     --hilos 2 --nombre js-1
```

### 4. Comandos útiles en la consola del servidor

```
nodos            # nodos conectados: lenguaje, hilos, velocidad
evaluar          # precisión y matriz de confusión
verificar        # compara Java vs Python vs Go vs JavaScript (debe dar OK)
entrenar 2       # entrenamiento distribuido de prueba
```

### 5. Cliente de chat

Ejecute **Run Maven → 3. Cliente de chat** en NetBeans, o `./cubes.sh cliente`. Credenciales por defecto: `alumno` / `uni2026`.

---

## Comandos del servidor

| Comando | Descripción |
|---|---|
| `nodos` | Estado de los nodos conectados |
| `estado` | Modelo, chat y clúster |
| `entrenar <épocas> [local]` | Entrenamiento distribuido o local |
| `evaluar [local]` | Precisión y matriz de confusión |
| `verificar` | Consistencia numérica entre lenguajes |
| `bench <tarea> <n> <destino>` | Mediciones de los 4 escenarios |
| `modo datos` / `modo modelo` | Estrategia de inferencia en el chat |
| `tta 1` / `tta 2` / `tta 10` | Vistas promediadas por imagen (TTA) |
| `esperar <n> [seg]` | Espera hasta que haya n nodos (útil en scripts) |
| `guardar` / `cargar` / `reiniciar` | Gestión de pesos en `modelos/modelo.ocw` |
| `silencio` / `verboso` | Verbosidad del log de consultas |
| `salir` | Cierra el servidor |

---

## Benchmarks (4 escenarios)

```bash
bench entrenar 2000 local:1              # Secuencial
bench entrenar 2000 local:4              # Paralelo
bench entrenar 2000 cluster Distribuido  # Distribuido (necesita nodos)
bench entrenar 2000 cluster ParDist      # Paralelo distribuido
```

Los resultados se acumulan en `resultados/benchmarks.csv`. Para generar gráficas:

```bash
./cubes.sh graficas
```

**Mediciones en Apple M4** (entrenar 2000 muestras):

| Escenario | Tiempo | Speedup |
|---|---|---|
| Secuencial | 1001 ms | ×1.00 |
| Paralelo 2h | 504 ms | ×1.99 |
| Paralelo 4h | 271 ms | ×3.69 |
| Paralelo 8h | 231 ms | ×4.34 |

---

## Datos y modelo incluido

El dataset real (gato, perro, panda) está en `datos/imagenes/` y el modelo preentrenado en `modelos/modelo.ocw` (77.17 % sin TTA, 78.17 % con TTA de 2 vistas). Para usar CIFAR-10 completo, descargue la versión binaria desde [cs.toronto.edu/~kriz/cifar-10-binary.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) y descomprima en `datos/`.

### Técnicas de regularización

| Técnica | Resultado |
|---|---|
| Sin augmentation, lr constante | 71.7 % de media |
| Con augmentation | 75.2 % (+3.5) |
| Augmentation + lr coseno | 77.2 % (+5.5) |
| + TTA 2 vistas | 78.17 % (+6.5) |

---

## Despliegue en clúster

```bash
# En la PC servidor (IP: 192.168.1.10)
./cubes.sh servidor

# En cada PC nodo
cubes.bat nodo-java   --servidor 192.168.1.10 --hilos 4 --nombre pc-java
python nodo_python/nodo.py --servidor 192.168.1.10 --hilos 4 --nombre pc-py
cd nodo_go && go run . --servidor 192.168.1.10 --hilos 4 --nombre pc-go
node nodo_js/nodo.js --servidor 192.168.1.10 --hilos 4 --nombre pc-js
```

Abra los puertos **TCP 5000** (nodos) y **6000** (clientes) en el firewall. Los nodos se reconectan automáticamente si el servidor se reinicia.

Para ejecutar en modo no interactivo (scripts de clúster):

```bash
java -cp build/classes pe.uni.cubes.servidor.ServidorCentral \
  --cmd "esperar 4 120;entrenar 5;evaluar;bench inferir 1000 cluster;salir"
```

---

## Prueba de estrés

```bash
./cubes.sh estres --servidor IP --clientes 50  --tx 40 --delayMin 0 --delayMax 100
./cubes.sh estres --servidor IP --clientes 200 --tx 20 --delayMin 0 --delayMax 20
./cubes.sh estres --servidor IP --clientes 400 --tx 10 --delayMin 0 --delayMax 0
```

Los resultados (throughput, latencia p50/p95/p99, errores 503) se guardan en `resultados/estres_resumen.csv`.

---

## Decisiones de diseño

- **Solo Java decodifica imágenes.** El servidor convierte JPG/PNG a 3×32×32 bytes con `ImageIO` y envía bytes (3 KB/imagen), no floats (12 KB).
- **Entrenamiento síncrono (parameter server).** El servidor envía pesos → reparte lote → recibe gradientes → promedia y aplica SGD. Matemáticamente equivalente al entrenamiento secuencial.
- **Reparto proporcional a la velocidad.** Cada muestra se asigna al nodo que terminaría antes. Python puro, ~50–100× más lento que Java o Go, recibe 0 muestras en lotes pequeños y no frena al resto.
- **Paralelismo interno por lenguaje:** Java usa `ExecutorService`; Go usa goroutines; Python usa `multiprocessing` (el GIL impide hilos reales); Node.js usa `worker_threads` con pesos compartidos en `SharedArrayBuffer`.
- **Precisión float32 en JavaScript.** Los pesos viven en `Float32Array` y cada operación se redondea con `Math.fround`, reproduciendo bit a bit la aritmética de Java. El comando `verificar` da diferencia 0.
- **Tolerancia a fallos.** Latidos cada 5 s; nodo sin respuesta durante 30 s se retira y su trabajo se reasigna.
- **Modo "modelo" (micro chunks).** Cada capa se divide por filtros/neuronas entre los nodos (`CONV_FWD` / `MLP_FWD`). Es más lento que el modo "datos" para una sola imagen (4 rondas de red), pero ilustra la distribución a nivel de micro chunk y el costo de comunicación (ley de Amdahl).

---

## Solución de problemas

| Problema | Solución |
|---|---|
| Maven no descarga plugins | Requiere internet una sola vez. Sin red, use `cubes.bat compilar` |
| Nodos no conectan desde otra PC | Verifique firewall (puerto 5000), IP correcta y misma red |
| `Modelo no entrenado` en el chat | Ejecute `entrenar N` o copie `modelos/modelo.ocw` |
| Nodo Python muy lento | Es lo esperado (sin NumPy). Use `--hilos` igual al número de núcleos |
| `bench ... cluster` sin nodos | Inicie los nodos y use `esperar <n>` antes |
| `node: command not found` | Instale Node.js ≥ 18 |
