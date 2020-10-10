package main 

import (
	"os"
	"fmt"
	"log"
	"strings"
	"io/ioutil"
	"encoding/json"
	"encoding/csv"
	"path/filepath"
)


type PeassResult struct {
	VersionCount   int `json:"versionCount"`
	ChangeCount    int `json:"changeCount"`
	TestCaseCount  int `json:"testcaseCount"`
	VersionChanges map[string]TestCaseChanges `json:"versionChanges"`
}

type TestCaseChanges struct {
	TestCaseChanges map[string][]Measurement  `json:"testcaseChanges"`
}

type Measurement struct {
	Diff string `json:"diff"`
	Method string `json:"method"`
	OldTime float32 `json:"oldTime"`
	ChangePercent  float32 `json:"changePercent"`
	TValue float32 `json:"tvalue"`
	VMS int `json:"vms"`
}

func ParsePeassResults() {
	dir := filepath.FromSlash("data/peass/")
	abs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal("Cannot read absolute filepath", err)
	}
	fmt.Printf("abs path: %v\n", abs)
	files := getFiles(abs)
	fmt.Printf("files: %v\n\n", files)
	for _, filename := range files {
		fmt.Println(filename)
		jsonFile, err := os.Open("data/peass/" + filename)
		if err != nil {
			fmt.Println(err)
		}
		fmt.Println("Successfuly opened json file...")
		defer jsonFile.Close()

		byteValue, err := ioutil.ReadAll(jsonFile)
		if err != nil {
			fmt.Println(err)
		}
		resFilename := strings.ReplaceAll(filename, ".json", ".csv")
		file, err := os.Create("results/" + resFilename)
		if err != nil {
			log.Fatal("Cannot create file", err)
		}
		defer file.Close()

		writer := csv.NewWriter(file)
		defer writer.Flush()

		var result PeassResult
		var data []string
		json.Unmarshal([]byte(byteValue), &result)
		for commit, v := range result.VersionChanges {
			for _, w := range v.TestCaseChanges {
				for _, j := range w {
					s := fmt.Sprintf("%v, %v, %v, %v, %v\n", commit, j.Diff, j.Method, j.OldTime, j.ChangePercent)
					data = append(data, s)
				}
			}
		}
		err = writer.Write(data)
		if err != nil {
			log.Fatal("Cannot write to file", err)
		}
	}
}

func getFiles(root string) []string {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if filepath.Ext(path) != ".json" {
			return nil
		}
		files = append(files, info.Name())
		return nil
	})
	if err != nil {
		panic(err)
	}
	return files
}
