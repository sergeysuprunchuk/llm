package mha

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"llm/pkg/lib"
	"testing"
)

func Test_Head_Forward(t *testing.T) {
	tests := []struct {
		head *Head
		input,
		output *mat.Dense
	}{
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			head: &Head{
				WQuery: mat.NewDense(2, 2, []float64{
					.2, -.1,
					.3, .4,
				}),
				WKey: mat.NewDense(2, 2, []float64{
					.4, .2,
					-.1, .5,
				}),
				WValue: mat.NewDense(2, 2, []float64{
					.5, -.2,
					.1, .3,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				.23, -.16,
				.2452, -.0279,
			}),
		},
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			head: &Head{
				WQuery: mat.NewDense(2, 2, []float64{
					-.3, .2,
					.1, -.4,
				}),
				WKey: mat.NewDense(2, 2, []float64{
					.2, -.3,
					.4, .1,
				}),
				WValue: mat.NewDense(2, 2, []float64{
					-.2, .4,
					.3, -.1,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				-.16, .22,
				-.0316, .1607,
			}),
		},
	}

	for i, test := range tests {
		output := test.head.Forward(test.input)

		for row := range lib.Rown(test.output) {
			grow := output.RawRowView(row)
			erow := test.output.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Head_Backward(t *testing.T) {
	tests := []struct {
		head *Head
		input,
		output,
		grad *mat.Dense
	}{
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			head: &Head{
				WQuery: mat.NewDense(2, 2, []float64{
					.2, -.1,
					.3, .4,
				}),
				WKey: mat.NewDense(2, 2, []float64{
					.4, .2,
					-.1, .5,
				}),
				WValue: mat.NewDense(2, 2, []float64{
					.5, -.2,
					.1, .3,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				.17, .11,
				-.19, .11,
			}),
			grad: mat.NewDense(2, 2, []float64{
				.0053, .0570,
				-.0597, .0082,
			}),
		},
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			head: &Head{
				WQuery: mat.NewDense(2, 2, []float64{
					-.3, .2,
					.1, -.4,
				}),
				WKey: mat.NewDense(2, 2, []float64{
					.2, -.3,
					.4, .1,
				}),
				WValue: mat.NewDense(2, 2, []float64{
					-.2, .4,
					.3, -.1,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				.14, -.07,
				-.04, .13,
			}),
			grad: mat.NewDense(2, 2, []float64{
				-.0264, .0377,
				.0300, -.0130,
			}),
		},
	}

	for i, test := range tests {
		test.head.Forward(test.input)
		output := test.head.Backward(test.output, 1)

		for row := range lib.Rown(test.grad) {
			grow := output.RawRowView(row)
			erow := test.grad.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Head_New(t *testing.T) {
	tests := []struct {
		icol, wcol int
	}{
		{
			icol: 4,
			wcol: 8,
		},
	}

	for i, test := range tests {
		head := NewHead(test.icol, test.wcol)

		if lib.Rown(head.WQuery) != lib.Rown(head.WKey) ||
			lib.Coln(head.WQuery) != lib.Coln(head.WKey) ||
			lib.Rown(head.WQuery) != lib.Rown(head.WValue) ||
			lib.Coln(head.WQuery) != lib.Coln(head.WValue) {
			t.Errorf("%d: матрицы разных размеров", i)
		}

		if lib.Rown(head.WQuery) != test.icol || lib.Coln(head.WQuery) != test.wcol {
			t.Errorf("%d: expected %dx%d, got %dx%d",
				i, test.icol, test.wcol, lib.Rown(head.WQuery), lib.Coln(head.WQuery))
		}
	}
}

func Test_Forward(t *testing.T) {
	tests := []struct {
		mha *MHA
		input,
		output *mat.Dense
	}{
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			mha: &MHA{
				Heads: []*Head{
					{
						WQuery: mat.NewDense(2, 2, []float64{
							.2, -.1,
							.3, .4,
						}),
						WKey: mat.NewDense(2, 2, []float64{
							.4, .2,
							-.1, .5,
						}),
						WValue: mat.NewDense(2, 2, []float64{
							.5, -.2,
							.1, .3,
						}),
					},
					{
						WQuery: mat.NewDense(2, 2, []float64{
							-.3, .2,
							.1, -.4,
						}),
						WKey: mat.NewDense(2, 2, []float64{
							.2, -.3,
							.4, .1,
						}),
						WValue: mat.NewDense(2, 2, []float64{
							-.2, .4,
							.3, -.1,
						}),
					},
				},
				WOutput: mat.NewDense(4, 2, []float64{
					.5, -.2,
					-.1, .4,
					.2, .1,
					-.3, .2,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				.033, -.082,
				.0708, -.0312,
			}),
		},
	}

	for i, test := range tests {
		output := test.mha.Forward(test.input)

		for row := range lib.Rown(test.output) {
			grow := output.RawRowView(row)
			erow := test.output.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}

func Test_Backward(t *testing.T) {
	tests := []struct {
		mha *MHA
		input,
		output,
		grad *mat.Dense
	}{
		{
			input: mat.NewDense(2, 2, []float64{
				.5, -.2,
				.4, .6,
			}),
			mha: &MHA{
				Heads: []*Head{
					{
						WQuery: mat.NewDense(2, 2, []float64{
							.2, -.1,
							.3, .4,
						}),
						WKey: mat.NewDense(2, 2, []float64{
							.4, .2,
							-.1, .5,
						}),
						WValue: mat.NewDense(2, 2, []float64{
							.5, -.2,
							.1, .3,
						}),
					},
					{
						WQuery: mat.NewDense(2, 2, []float64{
							-.3, .2,
							.1, -.4,
						}),
						WKey: mat.NewDense(2, 2, []float64{
							.2, -.3,
							.4, .1,
						}),
						WValue: mat.NewDense(2, 2, []float64{
							-.2, .4,
							.3, -.1,
						}),
					},
				},
				WOutput: mat.NewDense(4, 2, []float64{
					.5, -.2,
					-.1, .4,
					.2, .1,
					-.3, .2,
				}),
			},
			output: mat.NewDense(2, 2, []float64{
				.5, .4,
				-.3, .2,
			}),
			grad: mat.NewDense(2, 2, []float64{
				-.0211, .0948,
				-.0296, -.0048,
			}),
		},
	}

	for i, test := range tests {
		test.mha.Forward(test.input)
		output := test.mha.Backward(test.output, 1)

		for row := range lib.Rown(test.grad) {
			grow := output.RawRowView(row)
			erow := test.grad.RawRowView(row)

			if !floats.EqualApprox(grow, erow, 1e-2) {
				t.Errorf("%d %d: expected %v, got %v", i, row, erow, grow)
			}
		}
	}
}
