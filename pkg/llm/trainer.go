package llm

import (
	"gonum.org/v1/gonum/mat"
	"iter"
	"llm/pkg/bpe"
	"llm/pkg/dirreader"
	"llm/pkg/lib"
	"log"
)

const (
	eot = "</eot>"
	pad = "</pad>"
)

func Train(
	llm *LLM,
	dataset string,
	bpe *bpe.BPE,
	dropoutP,
	lr float64,
	saveIn string,
) {
	if !bpe.Has(eot) {
		panic("токена eot нет в словаре")
	}

	if !bpe.Has(pad) {
		panic("токена pad нет в словаре")
	}

	for index, exam := range examples(dataset, bpe, llm.CtxSize) {
		output := llm.Forward(exam.input, dropoutP)
		log.Printf("ошибка %.2f; пример %d\n",
			lib.CrossEntropy(output, exam.answer), index)
		output.Sub(output, exam.answer)
		llm.Backward(output, lr)
		if index%1000 == 0 {
			log.Println("сохранение")
			llm.Save(saveIn)
		}
	}
}

type example struct {
	input  []int
	answer *mat.Dense
}

func examples(src string, bpe *bpe.BPE, winsize int) iter.Seq2[int, example] {
	return func(yield func(int, example) bool) {
		var n int
		padind := bpe.GetInd(pad)

		for filename, data := range dirreader.Read(src) {
			inds := bpe.GetTextInds(string(data))
			inds = append(inds, bpe.GetInd(eot))

			for i, l := 0, len(inds); i+1 < l; i += winsize / 2 {
				n++

				var haspads bool
				for len(inds) < i+winsize+1 {
					inds = append(inds, padind)
					haspads = true
				}

				input := inds[i : i+winsize]

				if !yield(n, example{
					input:  input,
					answer: lib.HotEnc(inds[i+1:i+winsize+1], bpe.Len()),
				}) {
					return
				}

				if haspads {
					break
				}
			}

			log.Printf("изучил файл %s\n", filename)
		}
	}
}
