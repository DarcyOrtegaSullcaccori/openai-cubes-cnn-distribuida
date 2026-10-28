package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
)

const (
	Magic      = 0x4F43
	Version    = 1
	MaxPayload = 64 * 1024 * 1024

	Registro      = 1
	RegistroOK    = 2
	Latido        = 3
	SyncPesos     = 4
	Ack           = 5
	TareaEntrenar = 10
	Gradientes    = 11
	TareaInferir  = 12
	Predicciones  = 13
	ConvFwd       = 14
	ConvOut       = 15
	MlpFwd        = 16
	MlpOut        = 17
	Error         = 20
	ErrGeneral    = 1
)

type Trama struct {
	Tipo    int
	MsgID   int32
	Payload []byte
}

func LeerTrama(r *bufio.Reader) (*Trama, error) {
	cab := make([]byte, 12)
	if _, err := io.ReadFull(r, cab); err != nil {
		return nil, err
	}
	if binary.BigEndian.Uint16(cab[0:2]) != Magic {
		return nil, errors.New("MAGIC invalido")
	}
	if cab[2] != Version {
		return nil, fmt.Errorf("version no soportada: %d", cab[2])
	}
	largo := int32(binary.BigEndian.Uint32(cab[8:12]))
	if largo < 0 || largo > MaxPayload {
		return nil, fmt.Errorf("longitud invalida: %d", largo)
	}
	t := &Trama{Tipo: int(cab[3]), MsgID: int32(binary.BigEndian.Uint32(cab[4:8])), Payload: make([]byte, largo)}
	if _, err := io.ReadFull(r, t.Payload); err != nil {
		return nil, err
	}
	return t, nil
}

type Emisor struct {
	mu sync.Mutex
	w  *bufio.Writer
}

func (e *Emisor) Enviar(tipo int, msgID int32, payload []byte) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	cab := make([]byte, 12)
	binary.BigEndian.PutUint16(cab[0:2], Magic)
	cab[2] = Version
	cab[3] = byte(tipo)
	binary.BigEndian.PutUint32(cab[4:8], uint32(msgID))
	binary.BigEndian.PutUint32(cab[8:12], uint32(len(payload)))
	if _, err := e.w.Write(cab); err != nil {
		return err
	}
	if _, err := e.w.Write(payload); err != nil {
		return err
	}
	return e.w.Flush()
}

type Lector struct {
	b []byte
	p int
}

func (l *Lector) I32() int32 {
	v := int32(binary.BigEndian.Uint32(l.b[l.p:]))
	l.p += 4
	return v
}

func (l *Lector) Int() int { return int(l.I32()) }

func (l *Lector) Texto() string {
	n := l.Int()
	s := string(l.b[l.p : l.p+n])
	l.p += n
	return s
}

func (l *Lector) I32s(n int) []int32 {
	v := make([]int32, n)
	for i := range v {
		v[i] = l.I32()
	}
	return v
}

func (l *Lector) F32s(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.BigEndian.Uint32(l.b[l.p:]))
		l.p += 4
	}
	return v
}

func (l *Lector) Bytes(n int) []byte {
	v := l.b[l.p : l.p+n]
	l.p += n
	return v
}

type Escritor struct{ b []byte }

func (e *Escritor) I32(v int32) *Escritor {
	e.b = binary.BigEndian.AppendUint32(e.b, uint32(v))
	return e
}

func (e *Escritor) Int(v int) *Escritor { return e.I32(int32(v)) }

func (e *Escritor) F32(v float32) *Escritor {
	e.b = binary.BigEndian.AppendUint32(e.b, math.Float32bits(v))
	return e
}

func (e *Escritor) Texto(s string) *Escritor {
	e.Int(len(s))
	e.b = append(e.b, s...)
	return e
}

func (e *Escritor) F32s(v []float32) *Escritor {
	for _, x := range v {
		e.b = binary.BigEndian.AppendUint32(e.b, math.Float32bits(x))
	}
	return e
}
