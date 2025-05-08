package lib

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func Test_Rown(t *testing.T) {
	tests := []struct {
		m      *mat.Dense
		output int
	}{
		{
			m:      mat.NewDense(8, 1, nil),
			output: 8,
		},
	}

	for i, test := range tests {
		output := Rown(test.m)
		if output != test.output {
			t.Errorf("%d: expected %d, got %d", i, test.output, output)
		}
	}
}

func Test_Coln(t *testing.T) {
	tests := []struct {
		m      *mat.Dense
		output int
	}{
		{
			m:      mat.NewDense(1, 8, nil),
			output: 8,
		},
	}

	for i, test := range tests {
		output := Coln(test.m)
		if output != test.output {
			t.Errorf("%d: expected %d, got %d", i, test.output, output)
		}
	}
}

func Test_Relu(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output *mat.Dense
	}{
		{
			src: mat.NewDense(2, 3, []float64{
				-1, .1, 3.2,
				-.1, .5, 0,
			}),
			output: mat.NewDense(2, 3, []float64{
				-1 * Alpha, .1, 3.2,
				-.1 * Alpha, .5, 0,
			}),
		},
	}

	for i, test := range tests {
		var trg mat.Dense
		Relu(&trg, test.src)

		if !mat.Equal(&trg, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, trg)
		}
	}
}

func Test_ReluDeriv(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output *mat.Dense
	}{
		{
			src: mat.NewDense(2, 3, []float64{
				-1, .1, 3.2,
				-.1, .5, 0,
			}),
			output: mat.NewDense(2, 3, []float64{
				Alpha, 1, 1,
				Alpha, 1, 1,
			}),
		},
	}

	for i, test := range tests {
		var trg mat.Dense
		ReluDeriv(&trg, test.src)

		if !mat.Equal(&trg, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, trg)
		}
	}
}

func Test_Mask(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output *mat.Dense
	}{
		{
			src: mat.NewDense(3, 3, []float64{
				-1, .1, 3.2,
				-.1, .5, 0,
				.8, .3, -2,
			}),
			output: mat.NewDense(3, 3, []float64{
				-1, math.Inf(-1), math.Inf(-1),
				-.1, .5, math.Inf(-1),
				.8, .3, -2,
			}),
		},
	}

	for i, test := range tests {
		var trg mat.Dense
		Mask(&trg, test.src)

		if !mat.Equal(&trg, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, trg)
		}
	}
}

func Test_Softmax(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output *mat.Dense
	}{
		{
			src: mat.NewDense(2, 3, []float64{
				-1, .1, 3.2,
				-.1, .5, 0,
			}),
			output: mat.NewDense(2, 3, []float64{
				0.014146, 0.042497, 0.943356,
				0.254629, 0.463963, 0.281408,
			}),
		},
		{
			src: mat.NewDense(1, 2, []float64{
				800, 1300,
			}),
			output: mat.NewDense(1, 2, []float64{
				0, 1,
			}),
		},
	}

	for i, test := range tests {
		var trg mat.Dense
		Softmax(&trg, test.src)

		for row := range Rown(test.src) {
			grow := trg.RawRowView(row)
			erow := test.output.RawRowView(row)

			if !floats.EqualApprox(grow, erow, Epsilon) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Step(t *testing.T) {
	tests := []struct {
		trg, grad, output *mat.Dense
		lr                float64
	}{
		{
			trg: mat.NewDense(2, 3, []float64{
				18.3, -4.9, 3.2,
				-2.1, 0, 7.4,
			}),
			grad: mat.NewDense(2, 3, []float64{
				16.1, 5.3, -2,
				-.1, -.8, 3,
			}),
			output: mat.NewDense(2, 3, []float64{
				16.69, -5.430000000000001, 3.4000000000000004,
				-2.0900000000000003, 0.08000000000000002, 7.1000000000000005,
			}),
			lr: 0.1,
		},
	}

	for i, test := range tests {
		Step(test.trg, test.grad, test.lr)

		if !mat.Equal(test.trg, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, test.trg)
		}
	}
}

func Test_RowSums(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output []float64
	}{
		{
			src: mat.NewDense(2, 3, []float64{
				.1, .5, -.1,
				1, .1, -.2,
			}),
			output: []float64{.5, .9},
		},
	}

	for i, test := range tests {
		sums := RowSums(test.src)

		if !floats.EqualApprox(sums, test.output, Epsilon) {
			t.Errorf("%d: expected %v, got %v", i, test.output, sums)
		}
	}
}

func Test_SubVec(t *testing.T) {
	tests := []struct {
		src,
		output *mat.Dense
		vec []float64
	}{
		{
			src: mat.NewDense(2, 3, []float64{
				5, 7, 1,
				2, -1, 3,
			}),
			vec: []float64{3, -2},
			output: mat.NewDense(2, 3, []float64{
				2, 4, -2,
				4, 1, 5,
			}),
		},
	}

	for i, test := range tests {
		var trg mat.Dense
		SubVec(&trg, test.src, test.vec)

		if !mat.Equal(&trg, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, trg)
		}
	}
}

func Test_Concat(t *testing.T) {
	tests := []struct {
		trg,
		src,
		output *mat.Dense
	}{
		{
			trg: mat.NewDense(2, 1, []float64{
				3,
				-2,
			}),
			src: mat.NewDense(2, 2, []float64{
				-5, .1,
				7, 4,
			}),
			output: mat.NewDense(2, 3, []float64{
				3, -5, .1,
				-2, 7, 4,
			}),
		},
		{
			trg: &mat.Dense{},
			src: mat.NewDense(2, 2, []float64{
				-5, .1,
				7, 4,
			}),
			output: mat.NewDense(2, 2, []float64{
				-5, .1,
				7, 4,
			}),
		},
		{
			trg: mat.NewDense(2, 1, []float64{
				3,
				-2,
			}),
			src: &mat.Dense{},
			output: mat.NewDense(2, 1, []float64{
				3,
				-2,
			}),
		},
	}

	for i, test := range tests {
		Concat(test.trg, test.src)

		if !mat.Equal(test.output, test.trg) {
			t.Errorf("%d: expected %v, got %v", i, test.output, test.trg)
		}
	}
}

func Test_Split(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output []*mat.Dense
		n      int
	}{
		{
			src: mat.NewDense(2, 4, []float64{
				.3, -.1, 8, 1.9,
				0, 9.3, -.7, .8,
			}),
			n: 2,
			output: []*mat.Dense{
				mat.NewDense(2, 2, []float64{
					.3, -.1,
					0, 9.3,
				}),
				mat.NewDense(2, 2, []float64{
					8, 1.9,
					-.7, .8,
				}),
			},
		},
	}

	for i, test := range tests {
		output := Split(test.src, test.n)

		if len(output) != len(test.output) {
			t.Errorf("%d: len: expected %d, got %d", i, len(test.output), len(output))
		}

		for index := range test.output {
			if !mat.Equal(output[index], test.output[index]) {
				t.Errorf("%d: expected %v, got %v", i, test.output, output)
			}
		}
	}
}

func Test_DropoutMask(t *testing.T) {
	tests := []struct {
		r, c int
		p    float64
	}{
		{
			r: 8,
			c: 8,
			p: .2,
		},
		{
			r: 8,
			c: 8,
			p: .6,
		},
		{
			r: 8,
			c: 8,
			p: 0,
		},
		{
			r: 8,
			c: 8,
			p: 1,
		},
	}

	for i, test := range tests {
		mask := DropoutMask(test.r, test.c, test.p)

	loop:
		for row := range test.r {
			for col := range test.c {
				if mask.At(row, col) != 0 &&
					mask.At(row, col) != 1./(1.-test.p) {
					t.Errorf("%d: неправильная маска", i)
					break loop
				}
			}
		}

		t.Log(test.p, mat.Formatted(mask))
	}
}

func Test_ParamN(t *testing.T) {
	tests := []struct {
		src    *mat.Dense
		output int
	}{
		{
			src:    mat.NewDense(32, 32, nil),
			output: 1024,
		},
	}

	for i, test := range tests {
		output := ParamN(test.src)

		if output != test.output {
			t.Errorf("%d: expected %v, got %v", i, test.output, output)
		}
	}
}
