package llm

import (
	"gonum.org/v1/gonum/mat"
	"llm/pkg/lib"
	"llm/pkg/mha"
	"llm/pkg/mlp"
	"math"
)

type Layer struct {
	MHA *mha.MHA
	MLP *mlp.MLP
	mhaMask,
	mlpMask *mat.Dense
}

func (layer *Layer) Forward(
	input *mat.Dense,
	alphaMHA,
	alphaMLP,
	dropoutP float64) *mat.Dense {

	mhaOut := layer.MHA.Forward(input)

	mhaMask := lib.DropoutMask(lib.Rown(mhaOut), lib.Coln(mhaOut), dropoutP)
	mhaOut.MulElem(mhaOut, mhaMask)
	mhaOut.Scale(alphaMHA, mhaOut)
	mhaOut.Add(mhaOut, input)

	mlpOut := layer.MLP.Forward(mhaOut)
	mlpMask := lib.DropoutMask(lib.Rown(mlpOut), lib.Coln(mlpOut), dropoutP)
	mlpOut.MulElem(mlpOut, mlpMask)
	mlpOut.Scale(alphaMLP, mlpOut)
	mlpOut.Add(mlpOut, mhaOut)

	layer.mhaMask = mhaMask
	layer.mlpMask = mlpMask

	return mlpOut
}

func (layer *Layer) Backward(
	output *mat.Dense,
	alphaMHA,
	alphaMLP,
	lr float64) *mat.Dense {

	var mlpOut mat.Dense
	mlpOut.Scale(alphaMLP, output)
	mlpOut.MulElem(&mlpOut, layer.mlpMask)
	mhaOut := layer.MLP.Backward(&mlpOut, lr)

	mhaOut.Add(mhaOut, output)

	var input mat.Dense
	input.CloneFrom(mhaOut)

	mhaOut.Scale(alphaMHA, mhaOut)
	mhaOut.MulElem(mhaOut, layer.mhaMask)

	input.Add(&input, layer.MHA.Backward(mhaOut, lr))

	return &input
}

func (layer *Layer) ParamN() int {
	return layer.MHA.ParamN() +
		layer.MLP.ParamN()
}

func NewLayer(h, irow, icol int, wcol int) *Layer {
	return &Layer{
		MHA: mha.New(h, icol, wcol),
		MLP: mlp.New(irow, icol, icol*4, icol),
	}
}

type LLM struct {
	Embeds  *mat.Dense
	Pos     *mat.Dense
	Layers  []*Layer
	CtxSize int
	last    *mat.Dense
	indices []int
}

func (llm *LLM) Forward(indices []int, dropoutP float64) *mat.Dense {
	embeds := mat.NewDense(len(indices), lib.Coln(llm.Embeds), nil)
	for index, embindex := range indices {
		embeds.SetRow(index, llm.Embeds.RawRowView(embindex))
	}

	var input mat.Dense
	input.Add(embeds, llm.Pos)

	alphaMHA := math.Pow(2*float64(len(llm.Layers)), -.25)
	alphaMLP := math.Pow(8*float64(len(llm.Layers)), -.25)

	for _, layer := range llm.Layers {
		input = *layer.Forward(&input, alphaMHA, alphaMLP, dropoutP)
	}

	var output mat.Dense
	output.Mul(&input, llm.Embeds.T())
	lib.Softmax(&output, &output)

	llm.last = &input
	llm.indices = indices

	return &output
}

func (llm *LLM) Backward(output *mat.Dense, lr float64) {
	var layer mat.Dense
	layer.Mul(output, llm.Embeds)

	alphaMHA := math.Pow(2*float64(len(llm.Layers)), -.25)
	alphaMLP := math.Pow(8*float64(len(llm.Layers)), -.25)

	for index := len(llm.Layers) - 1; index >= 0; index-- {
		layer = *llm.Layers[index].
			Backward(&layer, alphaMHA, alphaMLP, lr)
	}

	var embeds mat.Dense
	embeds.Mul(llm.last.T(), output)
	embedsT := mat.DenseCopyOf(embeds.T())

	for index, embindex := range llm.indices {
		emb := embedsT.RawRowView(embindex)
		lay := layer.RawRowView(index)

		for j := range len(emb) {
			emb[j] += lay[j]
		}
	}

	lib.Step(llm.Pos, &layer, lr)
	lib.Step(llm.Embeds, embedsT, lr)
}

func (llm *LLM) ParamN() int {
	sum := lib.ParamN(llm.Embeds) +
		lib.ParamN(llm.Pos)

	for _, layer := range llm.Layers {
		sum += layer.ParamN()
	}

	return sum
}

func New(ctxsize, embrown, embcoln, l, h int) *LLM {
	layers := make([]*Layer, l)

	for index := range l {
		layers[index] = NewLayer(h, ctxsize, embcoln, embcoln/h)
	}

	return &LLM{
		Embeds:  lib.Xavier(embrown, embcoln),
		Pos:     lib.Xavier(ctxsize, embcoln),
		Layers:  layers,
		CtxSize: ctxsize,
	}
}
