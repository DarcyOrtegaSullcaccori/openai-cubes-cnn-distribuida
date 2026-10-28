package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type Estado struct {
	a       *Arquitectura
	p       []float32
	version int
}

type Motor struct {
	hilos  int
	estado atomic.Pointer[Estado]
}

func (m *Motor) actual() (*Estado, error) {
	s := m.estado.Load()
	if s == nil {
		return nil, errors.New("el nodo aun no tiene pesos (falta SYNC_PESOS)")
	}
	return s, nil
}

func (m *Motor) paralelo(total int, fn func(parte, ini, fin int)) int {
	partes := mini(m.hilos, total)
	if partes < 1 {
		partes = 1
	}
	var wg sync.WaitGroup
	for i := 0; i < partes; i++ {
		ini, fin := total*i/partes, total*(i+1)/partes
		wg.Add(1)
		go func(i, ini, fin int) {
			defer wg.Done()
			fn(i, ini, fin)
		}(i, ini, fin)
	}
	wg.Wait()
	return partes
}

func (m *Motor) Entrenar(imgs []byte, etq []int32) (grad []float32, perdida float32, aciertos int, err error) {
	s, err := m.actual()
	if err != nil {
		return nil, 0, 0, err
	}
	n, tam := len(etq), s.a.TamImagen()
	partes := mini(m.hilos, n)
	if partes < 1 {
		partes = 1
	}
	grads := make([][]float32, partes)
	perd := make([]float32, partes)
	acs := make([]int, partes)
	m.paralelo(n, func(parte, ini, fin int) {
		g := make([]float32, s.a.Total)
		e := NuevoEspacio(s.a)
		for i := ini; i < fin; i++ {
			l, ok := EntrenarMuestra(s.a, s.p, imgs[i*tam:(i+1)*tam], int(etq[i]), g, e)
			perd[parte] += l
			if ok {
				acs[parte]++
			}
		}
		grads[parte] = g
	})
	grad = grads[0]
	for k := 1; k < partes; k++ {
		for j := range grad {
			grad[j] += grads[k][j]
		}
	}
	for k := 0; k < partes; k++ {
		perdida += perd[k]
		aciertos += acs[k]
	}
	return grad, perdida, aciertos, nil
}

func (m *Motor) Inferir(imgs []byte, n int) ([]float32, error) {
	s, err := m.actual()
	if err != nil {
		return nil, err
	}
	tam, k := s.a.TamImagen(), s.a.K
	probs := make([]float32, n*k)
	m.paralelo(n, func(_, ini, fin int) {
		e := NuevoEspacio(s.a)
		for i := ini; i < fin; i++ {
			Predecir(s.a, s.p, imgs[i*tam:(i+1)*tam], probs[i*k:(i+1)*k], e)
		}
	})
	return probs, nil
}

func (m *Motor) Conv(capa, fIni, fFin int, entrada []float32) ([]float32, error) {
	s, err := m.actual()
	if err != nil {
		return nil, err
	}
	a := s.a
	C, H, W, F, wOff, bOff := a.C, a.H, a.W, a.F1, a.OW1, a.OB1
	if capa == 2 {
		C, H, W, F, wOff, bOff = a.F1, a.H2, a.W2, a.F2, a.OW2, a.OB2
	}
	if len(entrada) != C*H*W || fIni < 0 || fFin > F || fIni >= fFin {
		return nil, errors.New("CONV_FWD con dimensiones invalidas")
	}
	z := make([]float32, F*H*W)
	salida := make([]float32, F*(H/2)*(W/2))
	m.paralelo(fFin-fIni, func(_, ini, fin int) {
		ConvForward(entrada, C, H, W, s.p, wOff, bOff, fIni+ini, fIni+fin, z)
		ReluPool(z, fIni+ini, fIni+fin, H, W, salida, nil)
	})
	por := (H / 2) * (W / 2)
	return salida[fIni*por : fFin*por], nil
}

func (m *Motor) Mlp(capa, jIni, jFin int, entrada []float32) ([]float32, error) {
	s, err := m.actual()
	if err != nil {
		return nil, err
	}
	a := s.a
	nIn, nOut, wOff, bOff, relu := a.Plano, a.Oculta, a.OW3, a.OB3, true
	if capa == 2 {
		nIn, nOut, wOff, bOff, relu = a.Oculta, a.K, a.OW4, a.OB4, false
	}
	if len(entrada) != nIn || jIni < 0 || jFin > nOut || jIni >= jFin {
		return nil, errors.New("MLP_FWD con dimensiones invalidas")
	}
	salida := make([]float32, nOut)
	m.paralelo(jFin-jIni, func(_, ini, fin int) {
		DensaForward(entrada, nIn, s.p, wOff, bOff, jIni+ini, jIni+fin, salida, relu)
	})
	return salida[jIni:jFin], nil
}

func atender(em *Emisor, m *Motor, t *Trama) {
	l := &Lector{b: t.Payload}
	var tipoResp int
	out := &Escritor{}
	err := func() error {
		switch t.Tipo {
		case TareaEntrenar:
			l.I32()
			n := l.Int()
			etq := l.I32s(n)
			s, err := m.actual()
			if err != nil {
				return err
			}
			imgs := l.Bytes(n * s.a.TamImagen())
			t0 := time.Now()
			g, perdida, aciertos, err := m.Entrenar(imgs, etq)
			if err != nil {
				return err
			}
			out.Int(n).F32(perdida).Int(aciertos).Int(int(time.Since(t0).Milliseconds())).Int(len(g)).F32s(g)
			tipoResp = Gradientes
		case TareaInferir:
			n := l.Int()
			s, err := m.actual()
			if err != nil {
				return err
			}
			imgs := l.Bytes(n * s.a.TamImagen())
			t0 := time.Now()
			probs, err := m.Inferir(imgs, n)
			if err != nil {
				return err
			}
			out.Int(n).Int(s.a.K).Int(int(time.Since(t0).Milliseconds())).F32s(probs)
			tipoResp = Predicciones
		case ConvFwd:
			capa, fIni, fFin := l.Int(), l.Int(), l.Int()
			C, H, W := l.Int(), l.Int(), l.Int()
			r, err := m.Conv(capa, fIni, fFin, l.F32s(C*H*W))
			if err != nil {
				return err
			}
			out.Int(len(r)).F32s(r)
			tipoResp = ConvOut
		case MlpFwd:
			capa, jIni, jFin, nIn := l.Int(), l.Int(), l.Int(), l.Int()
			r, err := m.Mlp(capa, jIni, jFin, l.F32s(nIn))
			if err != nil {
				return err
			}
			out.Int(len(r)).F32s(r)
			tipoResp = MlpOut
		default:
			return fmt.Errorf("tipo no soportado: %d", t.Tipo)
		}
		return nil
	}()
	if err != nil {
		em.Enviar(Error, t.MsgID, (&Escritor{}).Int(ErrGeneral).Texto(err.Error()).b)
		return
	}
	em.Enviar(tipoResp, t.MsgID, out.b)
}

func sesion(servidor string, puerto int, nombre string, m *Motor) error {
	con, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", servidor, puerto), 5*time.Second)
	if err != nil {
		return err
	}
	defer con.Close()
	if tc, ok := con.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
	}
	r := bufio.NewReaderSize(con, 1<<16)
	em := &Emisor{w: bufio.NewWriterSize(con, 1<<16)}
	reg := (&Escritor{}).Texto(nombre).Texto("Go "+runtime.Version()[2:]).Int(runtime.NumCPU()).Int(m.hilos).Texto("CONV,MLP")
	if err := em.Enviar(Registro, 0, reg.b); err != nil {
		return err
	}
	t, err := LeerTrama(r)
	if err != nil {
		return err
	}
	if t.Tipo != RegistroOK {
		return errors.New("registro rechazado")
	}
	fmt.Printf("[NodoGo] Registrado en %s:%d con id %d\n", servidor, puerto, (&Lector{b: t.Payload}).I32())

	fin := make(chan struct{})
	defer close(fin)
	go func() {
		tk := time.NewTicker(5 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-fin:
				return
			case <-tk.C:
				if em.Enviar(Latido, 0, nil) != nil {
					return
				}
			}
		}
	}()

	for {
		t, err := LeerTrama(r)
		if err != nil {
			return err
		}
		switch t.Tipo {
		case SyncPesos:
			l := &Lector{b: t.Payload}
			version := l.Int()
			a := NuevaArquitectura(l.Int(), l.Int(), l.Int(), l.Int(), l.Int(), l.Int(), l.Int())
			p := l.F32s(l.Int())
			m.estado.Store(&Estado{a: a, p: p, version: version})
			em.Enviar(Ack, t.MsgID, (&Escritor{}).Int(version).b)
		case Latido:
		default:
			go atender(em, m, t)
		}
	}
}

func main() {
	host, _ := os.Hostname()
	servidor := flag.String("servidor", "localhost", "host del servidor central")
	puerto := flag.Int("puerto", 5000, "puerto de nodos del servidor")
	hilos := flag.Int("hilos", runtime.NumCPU(), "goroutines de calculo (H)")
	nombre := flag.String("nombre", "go-"+host, "nombre del nodo")
	flag.Parse()
	m := &Motor{hilos: maxi(1, *hilos)}
	fmt.Printf("[NodoGo] %s | hilos=%d | nucleos=%d | %s\n", *nombre, m.hilos, runtime.NumCPU(), runtime.Version())
	for {
		err := sesion(*servidor, *puerto, *nombre, m)
		fmt.Printf("[NodoGo] Conexion perdida/no disponible (%v). Reintento en 3 s...\n", err)
		time.Sleep(3 * time.Second)
	}
}
