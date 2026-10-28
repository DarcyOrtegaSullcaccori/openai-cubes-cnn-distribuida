package main

import "math"

type Arquitectura struct {
	C, H, W, F1, F2, Oculta, K                    int
	H2, W2, H4, W4, Plano                         int
	OW1, OB1, OW2, OB2, OW3, OB3, OW4, OB4, Total int
}

func NuevaArquitectura(c, h, w, f1, f2, oculta, k int) *Arquitectura {
	a := &Arquitectura{C: c, H: h, W: w, F1: f1, F2: f2, Oculta: oculta, K: k}
	a.H2, a.W2, a.H4, a.W4 = h/2, w/2, h/4, w/4
	a.Plano = f2 * a.H4 * a.W4
	a.OW1 = 0
	a.OB1 = a.OW1 + f1*c*9
	a.OW2 = a.OB1 + f1
	a.OB2 = a.OW2 + f2*f1*9
	a.OW3 = a.OB2 + f2
	a.OB3 = a.OW3 + oculta*a.Plano
	a.OW4 = a.OB3 + oculta
	a.OB4 = a.OW4 + k*oculta
	a.Total = a.OB4 + k
	return a
}

func (a *Arquitectura) TamImagen() int { return a.C * a.H * a.W }

type Espacio struct {
	x, z1, a1, z2, a2, z3, a3, o, prob []float32
	i1, i2                             []int
	dO, dA3, dZ3, dA2, dZ2, dA1, dZ1   []float32
}

func NuevoEspacio(a *Arquitectura) *Espacio {
	e := &Espacio{}
	e.x = make([]float32, a.C*a.H*a.W)
	e.z1 = make([]float32, a.F1*a.H*a.W)
	e.a1 = make([]float32, a.F1*a.H2*a.W2)
	e.i1 = make([]int, len(e.a1))
	e.z2 = make([]float32, a.F2*a.H2*a.W2)
	e.a2 = make([]float32, a.Plano)
	e.i2 = make([]int, len(e.a2))
	e.z3 = make([]float32, a.Oculta)
	e.a3 = make([]float32, a.Oculta)
	e.o = make([]float32, a.K)
	e.prob = make([]float32, a.K)
	e.dO = make([]float32, a.K)
	e.dA3 = make([]float32, a.Oculta)
	e.dZ3 = make([]float32, a.Oculta)
	e.dA2 = make([]float32, a.Plano)
	e.dZ2 = make([]float32, len(e.z2))
	e.dA1 = make([]float32, len(e.a1))
	e.dZ1 = make([]float32, len(e.z1))
	return e
}

func Normalizar(img []byte, x []float32) {
	for i := range x {
		x[i] = float32(img[i])/255 - 0.5
	}
}

func forward(a *Arquitectura, p []float32, e *Espacio) {
	ConvForward(e.x, a.C, a.H, a.W, p, a.OW1, a.OB1, 0, a.F1, e.z1)
	ReluPool(e.z1, 0, a.F1, a.H, a.W, e.a1, e.i1)
	ConvForward(e.a1, a.F1, a.H2, a.W2, p, a.OW2, a.OB2, 0, a.F2, e.z2)
	ReluPool(e.z2, 0, a.F2, a.H2, a.W2, e.a2, e.i2)
	DensaForward(e.a2, a.Plano, p, a.OW3, a.OB3, 0, a.Oculta, e.z3, false)
	for j := range e.z3 {
		if e.z3[j] > 0 {
			e.a3[j] = e.z3[j]
		} else {
			e.a3[j] = 0
		}
	}
	DensaForward(e.a3, a.Oculta, p, a.OW4, a.OB4, 0, a.K, e.o, false)
	Softmax(e.o, e.prob)
}

func Predecir(a *Arquitectura, p []float32, img []byte, probs []float32, e *Espacio) {
	Normalizar(img, e.x)
	forward(a, p, e)
	copy(probs, e.prob)
}

func limpiar(v []float32) {
	for i := range v {
		v[i] = 0
	}
}

func EntrenarMuestra(a *Arquitectura, p []float32, img []byte, etiqueta int, g []float32, e *Espacio) (float32, bool) {
	Normalizar(img, e.x)
	forward(a, p, e)
	mejor := 0
	for i := 1; i < a.K; i++ {
		if e.prob[i] > e.prob[mejor] {
			mejor = i
		}
	}
	perdida := float32(-math.Log(float64(e.prob[etiqueta]) + 1e-9))
	for i := 0; i < a.K; i++ {
		e.dO[i] = e.prob[i]
		if i == etiqueta {
			e.dO[i] -= 1
		}
	}
	limpiar(e.dA3)
	DensaBackward(e.a3, a.Oculta, e.dO, a.K, p, a.OW4, a.OB4, g, e.dA3)
	for j := range e.dZ3 {
		if e.z3[j] > 0 {
			e.dZ3[j] = e.dA3[j]
		} else {
			e.dZ3[j] = 0
		}
	}
	limpiar(e.dA2)
	DensaBackward(e.a2, a.Plano, e.dZ3, a.Oculta, p, a.OW3, a.OB3, g, e.dA2)
	ReluPoolBackward(e.dA2, e.i2, e.z2, a.F2, a.H2, a.W2, e.dZ2)
	limpiar(e.dA1)
	ConvBackward(e.a1, a.F1, a.H2, a.W2, e.dZ2, 0, a.F2, p, a.OW2, a.OB2, g, e.dA1)
	ReluPoolBackward(e.dA1, e.i1, e.z1, a.F1, a.H, a.W, e.dZ1)
	ConvBackward(e.x, a.C, a.H, a.W, e.dZ1, 0, a.F1, p, a.OW1, a.OB1, g, nil)
	return perdida, mejor == etiqueta
}
