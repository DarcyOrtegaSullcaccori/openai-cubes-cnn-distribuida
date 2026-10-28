package main

import "math"

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func ConvForward(in []float32, C, H, W int, p []float32, wOff, bOff, fIni, fFin int, z []float32) {
	hw := H * W
	for f := fIni; f < fFin; f++ {
		zo := f * hw
		b := p[bOff+f]
		for i := 0; i < hw; i++ {
			z[zo+i] = b
		}
		for c := 0; c < C; c++ {
			io := c * hw
			wo := wOff + (f*C+c)*9
			for ky := 0; ky < 3; ky++ {
				dy := ky - 1
				y0, y1 := maxi(0, -dy), mini(H, H-dy)
				for kx := 0; kx < 3; kx++ {
					dx := kx - 1
					x0, x1 := maxi(0, -dx), mini(W, W-dx)
					wv := p[wo+ky*3+kx]
					for y := y0; y < y1; y++ {
						zr := zo + y*W
						ir := io + (y+dy)*W + dx
						for x := x0; x < x1; x++ {
							z[zr+x] += wv * in[ir+x]
						}
					}
				}
			}
		}
	}
}

func ReluPool(z []float32, fIni, fFin, H, W int, out []float32, idx []int) {
	H2, W2, hw := H/2, W/2, H*W
	for f := fIni; f < fFin; f++ {
		for py := 0; py < H2; py++ {
			for px := 0; px < W2; px++ {
				base := f*hw + (2*py)*W + 2*px
				m := base
				v := z[base]
				if z[base+1] > v {
					v, m = z[base+1], base+1
				}
				if z[base+W] > v {
					v, m = z[base+W], base+W
				}
				if z[base+W+1] > v {
					v, m = z[base+W+1], base+W+1
				}
				o := f*H2*W2 + py*W2 + px
				if v > 0 {
					out[o] = v
				} else {
					out[o] = 0
				}
				if idx != nil {
					idx[o] = m
				}
			}
		}
	}
}

func ReluPoolBackward(dOut []float32, idx []int, z []float32, F, H, W int, dz []float32) {
	for i := 0; i < F*H*W; i++ {
		dz[i] = 0
	}
	n := F * (H / 2) * (W / 2)
	for o := 0; o < n; o++ {
		i := idx[o]
		if z[i] > 0 {
			dz[i] += dOut[o]
		}
	}
}

func ConvBackward(in []float32, C, H, W int, dz []float32, fIni, fFin int, p []float32, wOff, bOff int, g, dIn []float32) {
	hw := H * W
	for f := fIni; f < fFin; f++ {
		zo := f * hw
		var sb float32
		for i := 0; i < hw; i++ {
			sb += dz[zo+i]
		}
		g[bOff+f] += sb
		for c := 0; c < C; c++ {
			io := c * hw
			wo := wOff + (f*C+c)*9
			for ky := 0; ky < 3; ky++ {
				dy := ky - 1
				y0, y1 := maxi(0, -dy), mini(H, H-dy)
				for kx := 0; kx < 3; kx++ {
					dx := kx - 1
					x0, x1 := maxi(0, -dx), mini(W, W-dx)
					wv := p[wo+ky*3+kx]
					var acc float32
					for y := y0; y < y1; y++ {
						zr := zo + y*W
						ir := io + (y+dy)*W + dx
						for x := x0; x < x1; x++ {
							d := dz[zr+x]
							acc += d * in[ir+x]
							if dIn != nil {
								dIn[ir+x] += wv * d
							}
						}
					}
					g[wo+ky*3+kx] += acc
				}
			}
		}
	}
}

func DensaForward(x []float32, nIn int, p []float32, wOff, bOff, jIni, jFin int, out []float32, relu bool) {
	for j := jIni; j < jFin; j++ {
		wo := wOff + j*nIn
		var s float32
		for i := 0; i < nIn; i++ {
			s += p[wo+i] * x[i]
		}
		s += p[bOff+j]
		if relu && s < 0 {
			s = 0
		}
		out[j] = s
	}
}

func DensaBackward(x []float32, nIn int, dOut []float32, nOut int, p []float32, wOff, bOff int, g, dIn []float32) {
	for j := 0; j < nOut; j++ {
		d := dOut[j]
		if d == 0 {
			continue
		}
		g[bOff+j] += d
		wo := wOff + j*nIn
		for i := 0; i < nIn; i++ {
			g[wo+i] += d * x[i]
		}
		if dIn != nil {
			for i := 0; i < nIn; i++ {
				dIn[i] += p[wo+i] * d
			}
		}
	}
}

func Softmax(z []float32, prob []float32) {
	m := z[0]
	for _, v := range z {
		if v > m {
			m = v
		}
	}
	var s float64
	for i, v := range z {
		e := math.Exp(float64(v - m))
		prob[i] = float32(e)
		s += e
	}
	for i := range prob {
		prob[i] = float32(float64(prob[i]) / s)
	}
}
