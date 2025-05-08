package mlp

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func Test_Layer_Forward(t *testing.T) {
	tests := []struct {
		input  *mat.Dense
		layer  *Layer
		output *mat.Dense
	}{
		{
			input: mat.NewDense(1, 3, []float64{2, 1, 4}),
			layer: &Layer{
				Weights: mat.NewDense(3, 2, []float64{
					3, 1,
					4, 7,
					0, 3,
				}),
				Bias: mat.NewDense(1, 2, []float64{.5, -.75}),
			},
			output: mat.NewDense(1, 2, []float64{10.5, 20.25}),
		},
	}

	for i, test := range tests {
		output := test.layer.Forward(test.input)
		if !mat.Equal(output, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, output)
		}
	}
}

func Test_Layer_Backward(t *testing.T) {
	tests := []struct {
		input   *mat.Dense
		layer   *Layer
		truth   *mat.Dense
		grad    *mat.Dense
		weights *mat.Dense
		bias    *mat.Dense
	}{
		{
			input: mat.NewDense(1, 3, []float64{2, 1, 4}),
			layer: &Layer{
				Weights: mat.NewDense(3, 2, []float64{
					3, 1,
					4, 7,
					0, 3,
				}),
				Bias: mat.NewDense(1, 2, []float64{.5, -.75}),
			},
			truth: mat.NewDense(1, 2, []float64{11, 19}),
			grad:  mat.NewDense(1, 3, []float64{-.25, 6.75, 3.75}),
			bias:  mat.NewDense(1, 2, []float64{1, -2}),
			weights: mat.NewDense(3, 2, []float64{
				4, -1.5,
				4.5, 5.75,
				2, -2,
			}),
		},
	}

	for i, test := range tests {
		output := test.layer.Forward(test.input)
		output.Sub(output, test.truth)
		grad := test.layer.Backward(output, 1)
		if !mat.Equal(grad, test.grad) {
			t.Errorf("%d: grad: expected %v, got %v", i, test.grad, grad)
		}

		if !mat.Equal(test.layer.Bias, test.bias) {
			t.Errorf("%d: bias: expected %v, got %v", i, test.bias, test.layer.Bias)
		}

		if !mat.Equal(test.layer.Weights, test.weights) {
			t.Errorf("%d: weights: expected %v, got %v", i, test.weights, test.layer.Weights)
		}
	}
}

func Test_Layer_New(t *testing.T) {
	tests := []struct {
		irow, icol, wcol int
	}{
		{
			irow: 2,
			icol: 4,
			wcol: 8,
		},
	}

	for i, test := range tests {
		layer := NewLayer(test.irow, test.icol, test.wcol)
		row, col := layer.Bias.Dims()
		if test.irow != row || test.wcol != col {
			t.Errorf("%d: bias: expected %dx%d, got %dx%d", i, test.irow, test.wcol, row, col)
		}

		row, col = layer.Weights.Dims()
		if test.icol != row || test.wcol != col {
			t.Errorf("%d: weights: expected %dx%d, got %dx%d", i, test.icol, test.wcol, row, col)
		}
	}
}

func Test_Forward(t *testing.T) {
	tests := []struct {
		input  *mat.Dense
		mlp    *MLP
		output *mat.Dense
	}{
		{
			input: mat.NewDense(1, 3, []float64{5, 7, 8}),
			mlp: &MLP{Layers: []*Layer{
				{
					Weights: mat.NewDense(3, 2, []float64{
						-3, 8,
						5, 7,
						-1, 0,
					}),
					Bias: mat.NewDense(1, 2, []float64{.5, -1}),
				},
				{
					Weights: mat.NewDense(2, 1, []float64{
						2,
						-3,
					}),
					Bias: mat.NewDense(1, 1, []float64{4}),
				},
			}},
			output: mat.NewDense(1, 1, []float64{-235}),
		},

		{
			input: mat.NewDense(1, 3, []float64{2, -3, 1}),
			mlp: &MLP{Layers: []*Layer{
				{
					Weights: mat.NewDense(3, 2, []float64{
						3, -2,
						-1, 4,
						2, 0,
					}),
					Bias: mat.NewDense(1, 2, []float64{1, -2}),
				},
				{
					Weights: mat.NewDense(2, 1, []float64{
						-1,
						5,
					}),
					Bias: mat.NewDense(1, 1, []float64{3}),
				},
			}},
			output: mat.NewDense(1, 1, []float64{-9.9}),
		},
	}

	for i, test := range tests {
		output := test.mlp.Forward(test.input)
		if !mat.Equal(output, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, output)
		}
	}
}

func Test_Backward(t *testing.T) {
	tests := []struct {
		input  *mat.Dense
		mlp    *MLP
		output *mat.Dense
		truth  *mat.Dense
	}{
		{
			input: mat.NewDense(1, 3, []float64{2, -3, 1}),
			mlp: &MLP{Layers: []*Layer{
				{
					Weights: mat.NewDense(3, 2, []float64{
						3, -2,
						-1, 4,
						2, 0,
					}),
					Bias: mat.NewDense(1, 2, []float64{1, -2}),
				},
				{
					Weights: mat.NewDense(2, 1, []float64{
						-1,
						5,
					}),
					Bias: mat.NewDense(1, 1, []float64{3}),
				},
			}},
			output: mat.NewDense(1, 3, []float64{43.09, -16.68, 27.8}),
			truth:  mat.NewDense(1, 1, []float64{4}),
		},
	}

	for i, test := range tests {
		pred := test.mlp.Forward(test.input)
		pred.Sub(pred, test.truth)
		output := test.mlp.Backward(pred, 1)

		if !mat.Equal(output, test.output) {
			t.Errorf("%d: expected %v, got %v", i, test.output, output)
		}
	}
}

func Test_New(t *testing.T) {
	tests := []struct {
		irow, icol int
		wcoln      []int
	}{
		{
			irow:  2,
			icol:  4,
			wcoln: []int{8, 16},
		},
	}

	for i, test := range tests {
		mlp := New(test.irow, test.icol, test.wcoln...)

		for index, layer := range mlp.Layers {
			row, col := layer.Bias.Dims()

			if test.irow != row || test.wcoln[index] != col {
				t.Errorf("%d: weights: expected %dx%d, got %dx%d", i, test.irow, test.wcoln[index], row, col)
			}

			row, col = layer.Weights.Dims()
			if test.icol != row || test.wcoln[index] != col {
				t.Errorf("%d: weights: expected %dx%d, got %dx%d", i, test.icol, test.wcoln[index], row, col)
			}

			test.icol = test.wcoln[index]
		}
	}
}
