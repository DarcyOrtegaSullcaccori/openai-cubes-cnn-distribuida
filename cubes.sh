#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
CMD="${1:-ayuda}"; shift || true

if [ -d target/classes/pe ]; then CP=target/classes; else CP=build/classes; fi

compilar() {
  mkdir -p build/classes
  find src/main/java -name '*.java' > build/fuentes.txt
  if javac --release 8 -version >/dev/null 2>&1; then
    javac --release 8 -Xlint:-options -encoding UTF-8 -d build/classes @build/fuentes.txt
  else
    javac -encoding UTF-8 -d build/classes @build/fuentes.txt
  fi
  echo "Compilado en build/classes"
}

necesita_clases() { [ -d "$CP/pe" ] || { compilar; CP=build/classes; }; }

case "$CMD" in
  compilar)    compilar ;;
  servidor)    necesita_clases; exec java -cp "$CP" pe.uni.cubes.servidor.ServidorCentral "$@" ;;
  nodo-java)   necesita_clases; exec java -cp "$CP" pe.uni.cubes.nodo.NodoJava "$@" ;;
  nodo-python) exec python3 nodo_python/nodo.py "$@" ;;
  nodo-js)     exec node nodo_js/nodo.js "$@" ;;
  nodo-go)     cd nodo_go && go build -o nodo_go . && exec ./nodo_go "$@" ;;
  cliente)     necesita_clases; exec java -cp "$CP" pe.uni.cubes.cliente.ClienteChat "$@" ;;
  estres)      necesita_clases; exec java -cp "$CP" pe.uni.cubes.herramientas.PruebaEstres "$@" ;;
  graficas)    necesita_clases; exec java -cp "$CP" pe.uni.cubes.herramientas.Graficas "$@" ;;
  sinteticos)  necesita_clases; exec java -cp "$CP" pe.uni.cubes.herramientas.GenerarSinteticos "$@" ;;
  *) echo "Uso: ./cubes.sh <compilar|servidor|nodo-java|nodo-python|nodo-js|nodo-go|cliente|estres|graficas|sinteticos> [argumentos]" ;;
esac
