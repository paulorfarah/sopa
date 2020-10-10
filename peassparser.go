package main

import (
	"os"
	"fmt"
	"log"
	"io/ioutil"
	"encoding/json"
	"encoding/csv"
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

func main() {
	jsonFile, err := os.Open("data/peass/commons-compress.json")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfuly opened json file...")
	defer jsonFile.Close()

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		fmt.Println(err)
	}

	file, err := os.Create("results/commons-compress.csv")
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
