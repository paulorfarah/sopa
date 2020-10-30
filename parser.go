package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type PeassResult struct {
	VersionCount   int                        `json:"versionCount"`
	ChangeCount    int                        `json:"changeCount"`
	TestCaseCount  int                        `json:"testcaseCount"`
	VersionChanges map[string]TestCaseChanges `json:"versionChanges"`
}

type TestCaseChanges struct {
	TestCaseChanges map[string][]Measurement `json:"testcaseChanges"`
}

type Measurement struct {
	Diff          string  `json:"diff"`
	Method        string  `json:"method"`
	OldTime       float64 `json:"oldTime"`
	ChangePercent float64 `json:"changePercent"`
	TValue        float64 `json:"tvalue"`
	VMS           int     `json:"vms"`
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
		}
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
		var data [][]string
		s := []string{"commit", "method", "oldTime", "currTime", "diffTime", "changePercent"}
		data = append(data, s)
		json.Unmarshal([]byte(byteValue), &result)
		for commit, v := range result.VersionChanges {
			for _, w := range v.TestCaseChanges {
				for _, j := range w {
					currTime := j.OldTime + (j.OldTime * (j.ChangePercent / float64(100)))
					diffTime := currTime - j.OldTime
					s = []string{commit, j.Method, fmt.Sprintf("%f", j.OldTime), fmt.Sprintf("%f", currTime), fmt.Sprintf("%f", diffTime), fmt.Sprintf("%f", j.ChangePercent)}
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

type HadoopResult struct {
	Commit       string
	Class        string
	ResponseTime []float64
	CpuUsage     []float64
	MemoryUsage  []float64
}

func ParseHadoopResults() {
	var res = make(map[string]map[string]float64)
	infile, err := os.Open("data/hadoop/hadoop.csv")
	if err != nil {
		fmt.Println(err)
	}
	defer infile.Close()

	lines, err := csv.NewReader(infile).ReadAll()
	if err != nil {
		fmt.Println(err)
	}
	outfile, err := os.Create("results/sum/hadoop.csv")
	if err != nil {
		log.Fatal("Cannot create hadoop results file", err)
	}
	defer outfile.Close()

	writer := csv.NewWriter(outfile)
	defer writer.Flush()

	var commits []string
	for _, line := range lines {
		commit := line[0]
		method := line[1]
		_, ok := res[commit]
		if ok == false {
			res[commit] = make(map[string]float64)
		}
		sum := float64(0)
		for i := 2; i < 32; i++ {
			val, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				fmt.Println("### ERROR: Cannot read value", err)
			}
			sum += val
		}
		avg := sum / float64(30)
		res[commit][method] = avg
		commits = append(commits, commit)
	}
	prevCommits := GetPreviousCommits("https://github.com/apache/hadoop", "hadoop", commits)
	// sumRespTime := make(map[string]float64)
	for commit, mapMethod := range res {
		// current commit
		sum := float64(0)
		methods := []string{}
		for methodName, methodTime := range mapMethod {
			sum += methodTime
			methods = append(methods, methodName)
		}
		// sumRespTime[commit] = sum
		// fmt.Println(">>> ", commit, sum)

		//previous commit
		prevCommit := prevCommits[commit]
		sumPrev := float64(0)
		methodsPrev := []string{}
		for methodName, methodTime := range res[prevCommit] {
			sumPrev += methodTime
			methodsPrev = append(methods, methodName)
		}
		fmt.Printf("curr: %s sum: %f -  prev: %s sum: %f\n", commit, sum, prevCommit, sumPrev)
		fmt.Printf("methodsCurr: %v\n", methods)
		fmt.Printf("methodsPrev: %v\n", methodsPrev)
		sumStr := fmt.Sprintf("%f", sum)
		sumPrevStr := fmt.Sprintf("%f", sumPrev)
		var row = []string{commit, sumPrevStr, sumStr}
		writer.Write(row)
	}

}

func getFiles(root string, fileExt string) []string {
	var files []string
	items, err := ioutil.ReadDir(root)
	if err != nil {
		panic(err)
	}
	for _, file := range items {
		if !file.IsDir() && filepath.Ext(file.Name()) == fileExt {
			files = append(files, file.Name())
		}
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

	for _, filename := range files {
		mapOldTime := make(map[string]float64)
		mapNewTime := make(map[string]float64)
		csvfile, err := os.Open("results/" + filename)
		if err != nil {
			log.Fatal("Cannot open file ", err)
		}

		// Parse the file
		r := csv.NewReader(csvfile)
		// Iterate through the records
		firstLine := true
		for {
			// Read each record from csv
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				fmt.Println("Cannot read row: ", filename, err)
			}
			if firstLine == true {
				firstLine = false
			} else {
				// fmt.Printf("row: %s\n", record)
				//commit,method,oldTime,currTime,diffTime,changePercent
				if len(record) > 3 {
					commit := record[0]
					oldTime := record[2]
					currTime := record[3]
					// diffTime := record[4]
					// changePercent := record[5]

					v, err := strconv.ParseFloat(oldTime, 64)
					if err != nil {
						fmt.Println("Cannot parse float value oldTime: ", err)
					}
					// fmt.Println(commit, v)
					mapOldTime[commit] += v
					v, err = strconv.ParseFloat(currTime, 64)
					if err != nil {
						fmt.Println("Cannot parse float value currTime: ", err)
					}
					mapNewTime[commit] += v
				} else {
					fmt.Println("row has less than 3 fields: ", record)
				}
			}
		}
		// fmt.Println("############ mapOldTime")
		// for k, v := range mapOldTime {
		// 	fmt.Println(k, v)
		// }

		// fmt.Println("############ mapNewTime")
		// for k, v := range mapNewTime {
		// 	fmt.Println(k, v)
		// }

		//save summary file
		outfile, err := os.Create("results/sum/sum_" + filename)

		if err != nil {
			log.Fatalf("failed creating file: %s", err)
		}

		csvwriter := csv.NewWriter(outfile)

		for k, v := range mapOldTime {
			resOldTime := fmt.Sprintf("%f", v)
			resNewTime := fmt.Sprintf("%f", mapNewTime[k])
			var res = []string{k, resOldTime, resNewTime}
			err = csvwriter.Write(res)
			if err != nil {
				fmt.Println("Cannot write sum to file: ", err)
			}
		}

		csvwriter.Flush()

		csvfile.Close()
	}
}
