package mha

import (
	"gonum.org/v1/gonum/mat"
	"llm/pkg/lib"
	"math"
	"sync"
)

type Head struct {
	WQuery *mat.Dense
	WKey   *mat.Dense
	WValue *mat.Dense
	input,
	query,
	key,
	value,
	scores *mat.Dense
}

func (head *Head) Forward(input *mat.Dense) *mat.Dense {
	var query, key, value mat.Dense
	query.Mul(input, head.WQuery)
	key.Mul(input, head.WKey)
	value.Mul(input, head.WValue)

	sqrt := math.Sqrt(float64(lib.Coln(head.WKey)))

	var scores mat.Dense
	scores.Mul(&query, key.T())
	scores.Scale(1./sqrt, &scores)
	lib.Mask(&scores, &scores)
	lib.Softmax(&scores, &scores)

	var output mat.Dense
	output.Mul(&scores, &value)

	head.input = input
	head.query = &query
	head.key = &key
	head.value = &value
	head.scores = &scores

	return &output
}

func (head *Head) Backward(output *mat.Dense, lr float64) *mat.Dense {
	var softmax mat.Dense
	softmax.Mul(output, head.value.T())

	var mul mat.Dense
	mul.MulElem(&softmax, head.scores)

	var sub mat.Dense
	lib.SubVec(&sub, &softmax, lib.RowSums(&mul))

	sqrt := math.Sqrt(float64(lib.Coln(head.WKey)))

	var scores mat.Dense
	scores.MulElem(head.scores, &sub)
	scores.Scale(1./sqrt, &scores)

	var query, key, value mat.Dense
	query.Mul(&scores, head.key)
	key.Mul(scores.T(), head.query)
	value.Mul(head.scores.T(), output)

	inputT := head.input.T()

	var wquery, wkey, wvalue mat.Dense
	wquery.Mul(inputT, &query)
	wkey.Mul(inputT, &key)
	wvalue.Mul(inputT, &value)

	var input, input2, input3 mat.Dense
	input.Mul(&query, head.WQuery.T())
	input2.Mul(&key, head.WKey.T())
	input3.Mul(&value, head.WValue.T())

	input.Add(&input, &input2)
	input.Add(&input, &input3)

	lib.Step(head.WQuery, &wquery, lr)
	lib.Step(head.WKey, &wkey, lr)
	lib.Step(head.WValue, &wvalue, lr)

	return &input
}

func (head *Head) ParamN() int {
	return lib.ParamN(head.WQuery) +
		lib.ParamN(head.WKey) +
		lib.ParamN(head.WValue)
}

func NewHead(icol, wcol int) *Head {
	return &Head{
		WQuery: lib.Xavier(icol, wcol),
		WKey:   lib.Xavier(icol, wcol),
		WValue: lib.Xavier(icol, wcol),
	}
}

type MHA struct {
	Heads   []*Head
	WOutput *mat.Dense
	concat  *mat.Dense
}

func (mha *MHA) Forward(input *mat.Dense) *mat.Dense {
	results := make([]*mat.Dense, len(mha.Heads))

	var wg sync.WaitGroup
	wg.Add(len(mha.Heads))
	for index := range mha.Heads {
		go func(index int) {
			results[index] = mha.Heads[index].Forward(input)
			wg.Done()
		}(index)
	}
	wg.Wait()

	var concat mat.Dense
	for _, res := range results {
		lib.Concat(&concat, res)
	}

	var output mat.Dense
	output.Mul(&concat, mha.WOutput)

	mha.concat = &concat

	return &output
}

func (mha *MHA) Backward(output *mat.Dense, lr float64) *mat.Dense {
	var concat mat.Dense
	concat.Mul(output, mha.WOutput.T())

	grads := lib.Split(&concat, len(mha.Heads))
	var input mat.Dense

	var mut sync.Mutex

	var wg sync.WaitGroup
	wg.Add(len(grads))
	for index := range grads {
		go func(index int) {
			defer wg.Done()

			res := mha.Heads[index].Backward(grads[index], lr)

			mut.Lock()
			defer mut.Unlock()

			if input.IsEmpty() {
				input.CloneFrom(res)
				return
			}

			input.Add(&input, res)
		}(index)
	}

	var woutput mat.Dense
	woutput.Mul(mha.concat.T(), output)
	lib.Step(mha.WOutput, &woutput, lr)

	wg.Wait()
	return &input
}

func (mha *MHA) ParamN() int {
	sum := lib.ParamN(mha.WOutput)

	for _, head := range mha.Heads {
		sum += head.ParamN()
	}

	return sum
}

func New(h, icol int, wcol int) *MHA {
	heads := make([]*Head, h)

	for index := range h {
		heads[index] = NewHead(icol, wcol)
	}

	return &MHA{
		Heads:   heads,
		WOutput: lib.Xavier(h*wcol, icol),
	}
}
