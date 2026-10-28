package pe.uni.cubes.servidor;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class ServidorCentral {

    private final RegistroNodos registro = new RegistroNodos();
    private final ModuloImagenes modulo;
    private final ServicioChat chat;

    private ServidorCentral(Configuracion cfg) {
        modulo = new ModuloImagenes(cfg, registro);
        chat = new ServicioChat(cfg, modulo);
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String config = "servidor.properties";
        String comandos = null;
        for (int i = 0; i + 1 < args.length; i += 2) {
            if (args[i].equals("--config")) {
                config = args[i + 1];
            } else if (args[i].equals("--cmd")) {
                comandos = args[i + 1];
            }
        }
        Configuracion cfg = new Configuracion(config);
        System.out.println("=====================================================");
        System.out.println("  OPEN AI CUBES - Servidor Central (CC4P1 2026-II)");
        System.out.println("=====================================================");
        ServidorCentral s = new ServidorCentral(cfg);
        s.modulo.cargarModelo();
        s.registro.iniciar(cfg.ent("puerto.nodos", 5000));
        s.chat.iniciar(cfg.ent("puerto.clientes", 6000));
        System.out.println("Escriba 'ayuda' para ver los comandos.");
        if (comandos != null) {
            for (String c : comandos.split(";")) {
                if (!s.ejecutar(c.trim())) {
                    System.exit(0);
                }
            }
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String linea;
        System.out.print("cubes> ");
        while ((linea = br.readLine()) != null) {
            if (!s.ejecutar(linea.trim())) {
                System.exit(0);
            }
            System.out.print("cubes> ");
        }
        Thread.currentThread().join();
    }

    private boolean ejecutar(String linea) {
        if (linea.isEmpty()) {
            return true;
        }
        String[] p = linea.split("\\s+");
        List<String> a = new ArrayList<String>();
        for (int i = 1; i < p.length; i++) {
            a.add(p[i]);
        }
        try {
            switch (p[0].toLowerCase()) {
                case "ayuda":
                case "help":
                    ayuda();
                    break;
                case "nodos":
                    listarNodos();
                    break;
                case "estado":
                    System.out.println(modulo.estado());
                    System.out.println(chat.estado());
                    System.out.println("Nodos: " + registro.vivos().size() + " | hilos remotos: " + registro.hilosTotales()
                            + " | lenguajes: " + registro.lenguajes());
                    break;
                case "entrenar": {
                    int ep = a.isEmpty() ? 1 : Integer.parseInt(a.get(0));
                    boolean local = a.contains("local");
                    int max = 0;
                    for (String x : a) {
                        if (x.startsWith("max=")) {
                            max = Integer.parseInt(x.substring(4));
                        }
                    }
                    modulo.entrenar(ep, local, max);
                    break;
                }
                case "evaluar": {
                    int max = 0;
                    for (String x : a) {
                        if (x.startsWith("max=")) {
                            max = Integer.parseInt(x.substring(4));
                        }
                    }
                    modulo.evaluar(a.contains("local"), max);
                    break;
                }
                case "verificar":
                    modulo.verificar();
                    break;
                case "modo":
                    if (!a.isEmpty() && (a.get(0).equals("datos") || a.get(0).equals("modelo"))) {
                        modulo.setModo(a.get(0));
                    }
                    System.out.println("Modo de inferencia del chat: " + modulo.getModo()
                            + "  (datos = imagen completa a un nodo | modelo = capas repartidas en micro chunks)");
                    break;
                case "tta":
                    if (!a.isEmpty()) {
                        modulo.setVistasTta(Integer.parseInt(a.get(0)));
                    }
                    System.out.println("TTA: " + modulo.getVistasTta() + " vista(s) por imagen en 'evaluar' y en el chat modo datos"
                            + "  (1 = sin TTA | 2 = + espejo | 10 = espejo y desplazamientos de 2 px)");
                    break;
                case "bench": {
                    if (a.size() < 3) {
                        System.out.println("Uso: bench <entrenar|inferir|modelo|unitaria> <n> <local:H|cluster> [etiqueta...]");
                        break;
                    }
                    StringBuilder et = new StringBuilder();
                    for (int i = 3; i < a.size(); i++) {
                        et.append(i > 3 ? " " : "").append(a.get(i));
                    }
                    modulo.benchmark(a.get(0), Integer.parseInt(a.get(1)), a.get(2), et.toString());
                    break;
                }
                case "esperar": {
                    int n = a.isEmpty() ? 1 : Integer.parseInt(a.get(0));
                    int seg = a.size() > 1 ? Integer.parseInt(a.get(1)) : 120;
                    long fin = System.currentTimeMillis() + seg * 1000L;
                    while (registro.vivos().size() < n && System.currentTimeMillis() < fin) {
                        Thread.sleep(200);
                    }
                    System.out.println("Nodos conectados: " + registro.vivos().size());
                    break;
                }
                case "dormir":
                    Thread.sleep(1000L * (a.isEmpty() ? 1 : Integer.parseInt(a.get(0))));
                    break;
                case "guardar":
                    modulo.guardarModelo();
                    break;
                case "cargar":
                    modulo.cargarModelo();
                    break;
                case "reiniciar":
                    modulo.reiniciar();
                    break;
                case "silencio":
                    chat.setVerboso(false);
                    System.out.println("Registro de cada consulta de chat desactivado ('verboso' para activarlo).");
                    break;
                case "verboso":
                    chat.setVerboso(true);
                    break;
                case "salir":
                case "exit":
                    System.out.println("Hasta luego.");
                    return false;
                default:
                    System.out.println("Comando desconocido. Escriba 'ayuda'.");
            }
        } catch (Exception ex) {
            System.out.println("[Error] " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
        }
        return true;
    }

    private void listarNodos() {
        List<ConexionNodo> ns = registro.vivos();
        if (ns.isEmpty()) {
            System.out.println("No hay nodos conectados. Inicie NodoJava, nodo.py, nodo_go o nodo.js.");
            return;
        }
        System.out.printf("%-4s %-22s %-16s %-7s %-6s %-8s %-10s %-9s %s%n", "ID", "NOMBRE", "LENGUAJE", "NUCLEOS",
                "HILOS", "EN VUELO", "TAREAS", "MUESTR/S", "DIRECCION");
        for (ConexionNodo n : ns) {
            System.out.printf(Locale.US, "%-4d %-22s %-16s %-7d %-6d %-8d %-10d %-9.1f %s%n", n.id, n.nombre,
                    n.lenguaje.length() > 16 ? n.lenguaje.substring(0, 16) : n.lenguaje, n.nucleos, n.hilos,
                    n.enVuelo.get(), n.tareasCompletadas.get(), n.velocidadMedida ? n.velocidad : 0.0, n.remoto());
        }
    }

    private void ayuda() {
        System.out.println(String.join("\n",
                "Comandos:",
                "  nodos                         lista los nodos conectados (lenguaje, hilos, carga)",
                "  estado                        estado del modelo, del chat y del cluster",
                "  entrenar <epocas> [local] [max=N]   entrena (distribuido entre nodos, o 'local' en el servidor)",
                "  evaluar [local] [max=N]       precision y matriz de confusion con el conjunto de prueba",
                "  verificar                     compara resultados de cada nodo (Java/Python/Go/JS) con Java local",
                "  modo <datos|modelo>           como resolver las consultas del chat",
                "  tta <1|2|10>                  vistas promediadas por imagen al evaluar y en el chat modo datos",
                "  bench <tarea> <n> <destino> [etiqueta]   mide tiempos y guarda en resultados/benchmarks.csv",
                "        tarea: entrenar | inferir | modelo | unitaria",
                "        destino: local:1 (Secuencial) | local:H (Paralelo) | cluster (Distribuido / Paralelo distribuido)",
                "  esperar <n> [seg]             espera a que se conecten n nodos (util en scripts --cmd)",
                "  guardar | cargar | reiniciar  manejo del archivo de pesos",
                "  silencio | verboso            oculta/muestra cada consulta del chat (util en pruebas de estres)",
                "  salir"));
    }
}
