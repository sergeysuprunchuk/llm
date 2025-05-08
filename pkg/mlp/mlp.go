package mlp

import (
	"gonum.org/v1/gonum/mat"
	"llm/pkg/lib"
)

type Layer struct {
	Weights *mat.Dense `json:"weights"`
	Bias    *mat.Dense `json:"bias"`

	input, output *mat.Dense
}

func (layer *Layer) Forward(input *mat.Dense) *mat.Dense {
	var output mat.Dense
	output.Mul(input, layer.Weights)
	output.Add(&output, layer.Bias)
	layer.input, layer.output = input, &output
	return &output
}

func (layer *Layer) Backward(output *mat.Dense, lr float64) *mat.Dense {
	var input, weights mat.Dense
	weights.Mul(layer.input.T(), output)
	input.Mul(output, layer.Weights.T())
	lib.Step(layer.Weights, &weights, lr)
	lib.Step(layer.Bias, output, lr)
	return &input
}

func (layer *Layer) ParamN() int {
	return lib.ParamN(layer.Weights) + lib.ParamN(layer.Bias)
}

func NewLayer(irow, icol, wcol int) *Layer {
	return &Layer{
		Weights: lib.He(icol, wcol),
		Bias:    mat.NewDense(irow, wcol, nil),
	}
}

type MLP struct {
	Layers []*Layer `json:"layers"`
}

func (mlp *MLP) Forward(input *mat.Dense) *mat.Dense {
	for index, layer := range mlp.Layers {
		input = layer.Forward(input)

		if index != len(mlp.Layers)-1 {
			var act mat.Dense
			lib.Relu(&act, input)
			input = &act
		}
	}

	return input
}

func (mlp *MLP) Backward(output *mat.Dense, lr float64) *mat.Dense {
	for index := len(mlp.Layers) - 1; index >= 0; index-- {
		output = mlp.Layers[index].Backward(output, lr)

		if index != 0 {
			var relu mat.Dense
			lib.ReluDeriv(&relu, mlp.Layers[index-1].output)
			output.MulElem(&relu, output)
		}
	}

	return output
}

func (mlp *MLP) ParamN() int {
	var sum int

	for _, layer := range mlp.Layers {
		sum += layer.ParamN()
	}

	return sum
}

func New(irow, icol int, wcoln ...int) *MLP {
	layers := make([]*Layer, len(wcoln))

	for index, wcol := range wcoln {
		layers[index] = NewLayer(irow, icol, wcol)
		icol = wcol
	}

	return &MLP{Layers: layers}
}
