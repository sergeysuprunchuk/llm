package dirreader

import (
	"os"
	"path/filepath"
	"testing"
)

func Test_Read(t *testing.T) {
	const root = ".\\TEMP"

	descr := map[string]map[string]string{
		"a0bb": {
			"46f5": "c19dfffb",
			"95c4": "c02b",
		},
		"1757": {
			"a90a": "4753",
		},
		"1757\\f3d5": {
			"e3be": "78d5e3817b0d",
		},
	}

	data := make(map[string]string)

	for dirname, dircont := range descr {
		err := os.MkdirAll(filepath.Join(root, dirname), os.ModePerm)
		if err != nil {
			panic(err)
		}

		for filename, filecont := range dircont {
			path := filepath.Join(root, dirname, filename)

			file, err := os.Create(path)
			if err != nil {
				panic(err)
			}

			_, err = file.Write([]byte(filecont))
			if err != nil {
				panic(err)
			}

			file.Close()

			data[path] = filecont
		}
	}

	for path, val := range Read(root) {
		real, ok := data[path]
		if !ok {
			t.Errorf("неожиданный файл %s", path)
		}

		if real != string(val) {
			t.Errorf("содержимое файлов отличается")
		}

		delete(data, path)

		t.Log(path, string(val))
	}

	if len(data) != 0 {
		t.Errorf("остались необработанные файлы")
	}

	err := os.RemoveAll(root)
	if err != nil {
		panic(err)
	}
}
