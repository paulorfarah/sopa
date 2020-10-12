package main 

import (
	"os"
	"fmt"
	"log"
	"strings"
	"strconv"
	"io/ioutil"
	"encoding/json"
	"encoding/csv"
	"path/filepath"
	"github.com/tobgu/qframe"
	"github.com/tobgu/qframe/config/groupby"
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
	OldTime float64 `json:"oldTime"`
	ChangePercent  float64 `json:"changePercent"`
	TValue float64 `json:"tvalue"`
	VMS int `json:"vms"`
}

func ParsePeassResults() {
	dir := filepath.FromSlash("data/peass/")
	abs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal("Cannot read absolute filepath", err)
	}
	files := getFiles(abs, ".json")
	for _, filename := range files {
		jsonFile, err := os.Open("data/peass/" + filename)
		if err != nil {
			fmt.Println(err)
		} //else {
			//fmt.Println("Successfuly opened json file...")
		//}
		defer jsonFile.Close()

		byteValue, err := ioutil.ReadAll(jsonFile)
		if err != nil {
			fmt.Println(err)
		}
		resFilename := strings.ReplaceAll(filename, ".json", ".csv")
		file, err := os.Create("results/peass_" + resFilename)
		if err != nil {
			log.Fatal("Cannot create file", err)
		}
		defer file.Close()

		writer := csv.NewWriter(file)
		defer writer.Flush()

		var result PeassResult
		var data [][]string
		s := []string{"commit", "method", "oldTime", "currTime", "diffTime", "changePercent"}
		data = append(data, s)
		json.Unmarshal([]byte(byteValue), &result)
		for commit, v := range result.VersionChanges {
			for _, w := range v.TestCaseChanges {
				for _, j := range w {
					currTime := j.OldTime + (j.OldTime * j.ChangePercent)
					diffTime := currTime - j.OldTime
					s = []string{commit, j.Method, strconv.FormatFloat(j.OldTime, 'g', 1, 64), strconv.FormatFloat(currTime, 'g', 1, 64), strconv.FormatFloat(diffTime, 'g', 1, 64), strconv.FormatFloat(j.ChangePercent, 'g', 1, 64)}
					data = append(data, s)
				}
			}
		}
		err = writer.WriteAll(data)
		if err != nil {
			log.Fatal("Cannot write to file", err)
		}
	}
}

func getFiles(root string, fileExt string) []string {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if filepath.Ext(path) != fileExt {
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

func SummarizeResults() {
	dir := filepath.FromSlash("results/")
	abs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal("Cannot read absolute filepath", err)
	}
	files := getFiles(abs, ".csv")

	floatSum := func(ts []float64) float64 {
	    var result float64
            for _, x := range ts {
	        result += x
	    }
            return result
	}

	for _, file := range files {
		csvfile, err := os.Open("results/" + file)
		if err != nil {
			    log.Fatal(err)
		    }
		f := qframe.ReadCSV(csvfile)

		f = f.GroupBy(groupby.Columns("commit")).Aggregate(qframe.Aggregation{Fn: floatSum, Column: "oldTime"})
		fmt.Println(f.Sort(qframe.Order{Column: "commit"}))

	}
}
