package dirreader

import (
	"iter"
	"os"
	"path/filepath"
)

func Read(src string) iter.Seq2[string, []byte] {
	return func(yield func(string, []byte) bool) {
		read(src, yield)
	}
}

func read(src string, yield func(string, []byte) bool) bool {
	dir, err := os.ReadDir(src)
	if err != nil {
		panic(err)
	}

	for _, entry := range dir {
		newsrc := filepath.Join(src, entry.Name())

		if entry.IsDir() {
			if !read(newsrc, yield) {
				return false
			}

			continue
		}

		data, err := os.ReadFile(newsrc)
		if err != nil {
			panic(err)
		}

		if !yield(newsrc, data) {
			return false
		}
	}

	return true
}
