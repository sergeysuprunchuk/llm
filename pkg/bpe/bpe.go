package bpe

import (
	"encoding/gob"
	"github.com/blevesearch/segment"
	"os"
	"strings"
	"unicode"
)

type BPE struct {
	val map[string]int
	inv []string
	eow,
	unk string
}

func (bpe *BPE) GetTextInds(text string) []int {
	text = strings.ToLower(text)

	seg := segment.NewWordSegmenterDirect([]byte(text))

	inds := make([]int, 0, 64)
	for seg.Segment() {
		runes := []rune(seg.Text())
		if len(runes) == 1 && unicode.IsSpace(runes[0]) {
			continue
		}

		inds = append(inds, bpe.GetWordInds(seg.Text())...)
	}

	return inds
}

func (bpe *BPE) GetWordInds(word string) []int {
	word += bpe.eow

	inds := make([]int, 0, 8)

loop:
	for i := 0; i < len(word); {
		for j := len(word); j > i; {
			substr := word[i:j]
			if bpe.Has(substr) {
				inds = append(inds, bpe.GetInd(substr))
				i = j
				continue loop
			}

			if j == len(word) {
				j -= len(bpe.eow)
				continue
			}
			j--
		}

		inds = append(inds, bpe.GetInd(bpe.unk))
		i++
	}

	return inds
}

func (bpe *BPE) Has(tok string) bool {
	_, ok := bpe.val[tok]
	return ok
}

func (bpe *BPE) GetInd(tok string) int { return bpe.val[tok] }

func (bpe *BPE) PrepareInv() {
	bpe.inv = make([]string, len(bpe.val))
	for tok, ind := range bpe.val {
		bpe.inv[ind] = tok
	}
}

/*
GetTok перед использованием необходимо вызвать метод PrepareInv,
если он не был вызван ранее.
*/
func (bpe *BPE) GetTok(ind int) string { return bpe.inv[ind] }

type data struct {
	Val map[string]int
	EOW,
	UNK string
}

func Load(src string) *BPE {
	var d data

	file, err := os.Open(src)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = gob.
		NewDecoder(file).
		Decode(&d)
	if err != nil {
		panic(err)
	}

	bpe := &BPE{
		val: d.Val,
		eow: d.EOW,
		unk: d.UNK,
	}

	if len(bpe.eow) == 0 {
		panic("отсутствует токен eow")
	}

	if !bpe.Has(bpe.eow) {
		panic("токена eow нет в словаре")
	}

	if len(bpe.unk) == 0 {
		panic("отсутствует токен unk")
	}

	if !bpe.Has(bpe.unk) {
		panic("токена unk нет в словаре")
	}

	return bpe
}

func (bpe *BPE) Save(trg string) {
	file, err := os.Create(trg)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = gob.
		NewEncoder(file).
		Encode(data{
			Val: bpe.val,
			EOW: bpe.eow,
			UNK: bpe.unk,
		})
	if err != nil {
		panic(err)
	}
}
