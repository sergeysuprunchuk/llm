package llm

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"llm/pkg/lib"
	"llm/pkg/mha"
	"llm/pkg/mlp"
	"testing"
)

var layer = func() *Layer {
	return &Layer{
		MHA: &mha.MHA{
			Heads: []*mha.Head{
				{
					WQuery: mat.NewDense(2, 2, []float64{
						.1, .2,
						.3, .4,
					}),
					WKey: mat.NewDense(2, 2, []float64{
						.1, .2,
						.3, .4,
					}),
					WValue: mat.NewDense(2, 2, []float64{
						.1, .2,
						.3, .4,
					}),
				},
			},
			WOutput: mat.NewDense(2, 2, []float64{
				.1, .2,
				.3, .4,
			}),
		},
		MLP: &mlp.MLP{
			Layers: []*mlp.Layer{
				{
					Weights: mat.NewDense(2, 4, []float64{
						.1, .2, .5, .7,
						.3, .4, .6, .8,
					}),
					Bias: mat.NewDense(2, 4, nil),
				},
				{
					Weights: mat.NewDense(4, 2, []float64{
						.1, .5,
						.2, .6,
						.3, .7,
						.4, .8,
					}),
					Bias: mat.NewDense(2, 2, nil),
				},
			},
		},
	}
}

func Test_Layer_Forward(t *testing.T) {
	tests := []struct {
		layer *Layer
		input,
		output *mat.Dense
		alphaMHA,
		alphaMLP float64
	}{
		{
			alphaMHA: .46,
			alphaMLP: .31,
			input: mat.NewDense(2, 2, []float64{
				.5, .7,
				1.0, 1.2,
			}),
			output: mat.NewDense(2, 2, []float64{
				.7984, 1.3396,
				1.5072, 2.3000,
			}),
			layer: layer(),
		},
	}

	for i, test := range tests {
		output := test.layer.Forward(test.input, test.alphaMHA, test.alphaMLP, 0)

		for row := range lib.Rown(test.output) {
			grow := output.RawRowView(row)
			erow := test.output.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Layer_Backward(t *testing.T) {
	tests := []struct {
		layer *Layer
		input,
		output,
		grad *mat.Dense
		alphaMHA,
		alphaMLP float64
	}{
		{
			alphaMHA: .46,
			alphaMLP: .31,
			input: mat.NewDense(2, 2, []float64{
				.5, .7,
				1.0, 1.2,
			}),
			output: mat.NewDense(2, 2, []float64{
				.1, .2,
				.3, .4,
			}),
			grad: mat.NewDense(2, 2, []float64{
				.2224, .3968,
				.5047, .6927,
			}),
			layer: layer(),
		},
	}

	for i, test := range tests {
		test.layer.Forward(test.input, test.alphaMHA, test.alphaMLP, 0)
		output := test.layer.Backward(test.output, test.alphaMHA, test.alphaMLP, 1)

		for row := range lib.Rown(test.grad) {
			grow := output.RawRowView(row)
			erow := test.grad.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Forward(t *testing.T) {
	tests := []struct {
		llm    *LLM
		input  []int
		output *mat.Dense
	}{
		{
			input: []int{0, 1},
			output: mat.NewDense(2, 3, []float64{
				.2662, .3284, .4053,
				.2008, .3125, .4865,
			}),
			llm: &LLM{
				Embeds: mat.NewDense(3, 2, []float64{
					.1, .2,
					.3, .4,
					.5, .6,
				}),
				Pos: mat.NewDense(2, 2, []float64{
					.05, .05,
					.1, .1,
				}),
				Layers: []*Layer{layer()},
			},
		},
	}

	for i, test := range tests {
		output := test.llm.Forward(test.input, 0)

		for row := range lib.Rown(test.output) {
			grow := output.RawRowView(row)
			erow := test.output.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}
