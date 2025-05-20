package bpe

import (
	"reflect"
	"testing"
)

const (
	eow = "eow"
	unk = "unk"
)

var val = map[string]int{}

func init() {
	toks := []string{
		eow,
		unk,
		"на" + eow,
		"друг", "ой" + eow,
		"день" + eow,
		"по", "утру" + eow,
		"в" + eow,
		"ожидании" + eow,
	}

	for index, tok := range toks {
		val[tok] = index
	}
}

var bpe = &BPE{
	val: val,
	eow: eow,
	unk: unk,
}

func Test_GetWordInds(t *testing.T) {
	tests := []struct {
		bpe  *BPE
		word string
		out  []int
	}{
		{
			bpe:  bpe,
			word: "другой",
			out:  []int{3, 4},
		},
		{
			bpe:  bpe,
			word: "ожидании",
			out:  []int{9},
		},
		{
			bpe:  bpe,
			word: "поутру",
			out:  []int{6, 7},
		},
		{
			bpe:  bpe,
			word: "по",
			out:  []int{6, 0},
		},
		{
			bpe:  bpe,
			word: "попоподругдругна",
			out:  []int{6, 6, 6, 3, 3, 2},
		},
	}

	for _, test := range tests {
		out := test.bpe.GetWordInds(test.word)

		if !reflect.DeepEqual(out, test.out) {
			t.Errorf("GetWordInds(%v)=%v, want %v", test.word, out, test.out)
		}
	}
}

func Test_GetTextInds(t *testing.T) {
	tests := []struct {
		bpe  *BPE
		text string
		out  []int
	}{
		{
			bpe:  bpe,
			text: "Другой день поутру, В Ожидании",
			out:  []int{3, 4, 5, 6, 7, 1, 0, 8, 9},
		},
		{
			bpe:  bpe,
			text: "день в ожидании",
			out:  []int{5, 8, 9},
		},
		{
			bpe:  bpe,
			text: "день в ожидании день в ожидании день в ожидании",
			out:  []int{5, 8, 9, 5, 8, 9, 5, 8, 9},
		},
	}

	for _, test := range tests {
		out := test.bpe.GetTextInds(test.text)

		if !reflect.DeepEqual(out, test.out) {
			t.Errorf("GetWordInds(%v)=%v, want %v", test.text, out, test.out)
		}
	}
}
