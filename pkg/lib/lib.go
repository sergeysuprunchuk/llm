package lib

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"math/rand"
)

const (
	Alpha   = .01
	Epsilon = 1e-6
)

func Rown(m *mat.Dense) int {
	rown, _ := m.Dims()
	return rown
}

func Coln(m *mat.Dense) int {
	_, coln := m.Dims()
	return coln
}

func Relu(trg, src *mat.Dense) {
	trg.Apply(func(_, _ int, val float64) float64 {
		if val >= 0 {
			return val
		}
		return val * Alpha
	}, src)
}

func ReluDeriv(trg, src *mat.Dense) {
	trg.Apply(func(_, _ int, val float64) float64 {
		if val >= 0 {
			return 1
		}
		return Alpha
	}, src)
}

func Mask(trg, src *mat.Dense) {
	trg.Apply(func(i, j int, val float64) float64 {
		if j > i {
			return math.Inf(-1)
		}
		return val
	}, src)
}

func Softmax(trg, src *mat.Dense) {
	sums := make([]float64, Rown(src))
	var maxs []float64

	trg.Apply(func(i, _ int, val float64) float64 {
		if len(maxs) <= i {
			maxs = append(maxs, floats.Max(src.RawRowView(i)))
		}

		exp := math.Exp(val - maxs[i])

		sums[i] += exp

		return exp
	}, src)

	trg.Apply(func(i, _ int, val float64) float64 {
		return val / sums[i]
	}, trg)
}

func Step(trg, grad *mat.Dense, lr float64) {
	var scale mat.Dense
	scale.Scale(lr, grad)
	trg.Sub(trg, &scale)
}

func RowSums(src *mat.Dense) []float64 {
	sums := make([]float64, Rown(src))

	for row := range Rown(src) {
		sums[row] = floats.Sum(src.RawRowView(row))
	}

	return sums
}

func SubVec(trg, src *mat.Dense, vec []float64) {
	trg.Apply(func(i, _ int, val float64) float64 {
		return val - vec[i]
	}, src)
}

func Concat(trg, src *mat.Dense) {
	trgRown, trgColn := trg.Dims()
	srcRown, srcColn := src.Dims()

	if srcRown == 0 || srcColn == 0 {
		return
	}

	if trgRown != 0 && trgRown != srcRown {
		panic("concat failed: trg rown and src rown")
	}

	newtrg := mat.NewDense(srcRown, trgColn+srcColn, nil)
	newtrg.Copy(trg)
	newtrg.Slice(0, srcRown, trgColn, trgColn+srcColn).(*mat.Dense).
		Copy(src)

	*trg = *newtrg
}

func Split(src *mat.Dense, n int) []*mat.Dense {
	if src.IsEmpty() {
		return nil
	}

	rown, coln := src.Dims()

	if coln%n != 0 {
		panic("split matrix has wrong dimensions")
	}

	partSize := coln / n

	mats := make([]*mat.Dense, n)

	for i := range n {
		mats[i] = src.Slice(0, rown, i*partSize, i*partSize+partSize).(*mat.Dense)
	}

	return mats
}

func DropoutMask(r, c int, p float64) *mat.Dense {
	if p < 0 {
		p = 0
	}

	if p > 1 {
		p = 1
	}

	mask := mat.NewDense(r, c, nil)

	mask.Apply(func(_, _ int, _ float64) float64 {
		if rand.Float64() < p {
			return 0
		}
		return 1. / (1. - p)
	}, mask)

	return mask
}

func ParamN(src *mat.Dense) int {
	rown, coln := src.Dims()
	return rown * coln
}

func Xavier(r, c int) *mat.Dense {
	data := make([]float64, r*c)

	limit := math.Sqrt(6) /
		math.Sqrt(float64(r)+float64(c))

	dist := distuv.Uniform{Min: -limit, Max: limit}
	for index := range data {
		data[index] = dist.Rand()
	}

	return mat.NewDense(r, c, data)
}

func He(r, c int) *mat.Dense {
	data := make([]float64, r*c)

	dist := distuv.Normal{
		Sigma: math.Sqrt(2. / float64(r)),
	}
	for index := range data {
		data[index] = dist.Rand()
	}

	return mat.NewDense(r, c, data)
}

func CrossEntropy(pred, ans *mat.Dense) float64 {
	var sum float64

	for row := range Rown(ans) {
		for col := range Coln(ans) {
			a := ans.At(row, col)
			if a == 0 {
				continue
			}

			sum += a * math.Log(max(1e-8, min(pred.At(row, col), 1.)))
		}
	}

	return -(sum / float64(Rown(ans)))
}

func HotEnc(inds []int, l int) *mat.Dense {
	m := mat.NewDense(len(inds), l, nil)

	for row, ind := range inds {
		m.Set(row, ind, 1)
	}

	return m
}
