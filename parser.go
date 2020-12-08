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

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
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
	/*
		read dataset from data folder
	*/
	var res = make(map[string]map[string]float64)
	infile, err := os.Open("data/hadoop/hadoop2.csv")
	if err != nil {
		fmt.Println(err)
	}
	defer infile.Close()

	lines, err := csv.NewReader(infile).ReadAll()
	if err != nil {
		fmt.Println(err)
	}
	outfile, err := os.Create("results/hadoop.csv")
	if err != nil {
		log.Fatal("Cannot create hadoop summary results file", err)
	}
	defer outfile.Close()

	writer := csv.NewWriter(outfile)
	defer writer.Flush()

	// var commits []string
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
		// commits = append(commits, commit)
	}

	var repo *git.Repository
	dir := "hadoop"
	url := "https://github.com/apache/hadoop"
	repo = CloneRepo(url, dir)
	// var prevCommits = make(map[string]string)
	// for _, hash := range commits {
	// 	parents := GetParentsFromCommit(repo, hash)
	// 	if len(parents) == 1 {
	// 		prevCommits[hash] = parents[0]
	// 	}
	// }
	// prevCommits := TraverseCommitsWithPrevious(repo, commits)
	sumRespTime := make(map[string]float64)
	for commit, mapMethod := range res {
		// current commit
		sum := float64(0)
		methods := []string{}
		for methodName, methodTime := range mapMethod {
			sum += methodTime
			methods = append(methods, methodName)
		}
		sumRespTime[commit] = sum
		// fmt.Println(">>> ", commit, sum)

		//previous commit
		// prevCommit := prevCommits[commit]
		prevCommit := GetParentCommit(repo, plumbing.NewHash(commit))
		sumPrev := float64(0)
		methodsPrev := []string{}
		for methodName, methodTime := range res[prevCommit] {
			sumPrev += methodTime
			methodsPrev = append(methods, methodName)
		}
		methodsDiff := slicesDiff(methods, methodsPrev)
		if len(methodsDiff) == 0 {
			fmt.Printf("curr: %s sum: %f -  prev: %s sum: %f\n", commit, sum, prevCommit, sumPrev)
			// fmt.Printf("methodsCurr: %v\n", methods)
			// fmt.Printf("methodsPrev: %v\n", methodsPrev)
			sumStr := fmt.Sprintf("%f", sum)
			sumPrevStr := fmt.Sprintf("%f", sumPrev)
			var row = []string{commit, sumPrevStr, sumStr}
			writer.Write(row)
		}
	}
}

// type TravisBuild struct {
// 	Build_id                   int       `json:"tr_build_id"`
// 	Job_id                     int       `json:"tr_job_id"`
// 	Build_number               int       `json:"tr_build_number"`
// 	Original_commit            string    `json:"tr_original_commit"`
// 	Log_lan                    string    `json:"tr_log_lan"`
// 	Log_status                 string    `json:"tr_log_status"`
// 	Log_setup_time             string    `json:"tr_log_setup_time"`
// 	Log_analyzer               string    `json:"tr_log_analyzer"`
// 	Log_frameworks             string    `json:"tr_log_frameworks"`
// 	Log_bool_tests_ran         bool      `json:"tr_log_bool_tests_ran"`
// 	Log_bool_tests_failed      bool      `json:"tr_log_bool_tests_failed"`
// 	Log_num_tests_ok           int       `json:"tr_log_num_tests_ok"`
// 	Log_num_tests_failed       int       `json:"tr_log_num_tests_failed"`
// 	Log_num_tests_run          int       `json:"tr_log_num_tests_run"`
// 	Log_num_tests_skipped      int       `json:"tr_log_num_tests_skipped"`
// 	Log_num_test_suites_run    int       `json:"tr_log_num_test_suites_run"`
// 	Log_num_test_suites_ok     int       `json:"tr_log_num_test_suites_run"`
// 	Log_num_test_suites_failed int       `json:"tr_log_num_test_suites_ok"`
// 	Log_tests                  []string  `json:"tr_log_tests"`            //": "com.squareup.okhttp.internal.spdy.Http20Draft09Test#com.squareup.okhttp.internal.spdy.SpdyConnectionTest#com.squareup.okhttp.internal.spdy.HpackDraft05Test#com.squareup.okhttp.internal.spdy.ByteArrayPoolTest#com.squareup.okhttp.internal.spdy.SettingsTest#com.squareup.okhttp.mockwebserver.CustomDispatcherTest#com.squareup.okhttp.mockwebserver.MockWebServerTest#com.squareup.okhttp.MediaTypeTest#com.squareup.okhttp.ConnectionPoolTest#com.squareup.okhttp.AsyncApiTest#com.squareup.okhttp.RequestTest#com.squareup.okhttp.DispatcherTest#com.squareup.okhttp.internal.tls.HostnameVerifierTest#com.squareup.okhttp.internal.StrictLineReaderTest#com.squareup.okhttp.internal.http.HttpOverSpdy3Test#com.squareup.okhttp.internal.http.HttpResponseCacheTest#com.squareup.okhttp.internal.http.HttpOverHttp20Draft09Test#com.squareup.okhttp.internal.http.StatusLineTest#com.squareup.okhttp.internal.http.RouteSelectorTest#com.squareup.okhttp.internal.http.HeadersTest#com.squareup.okhttp.internal.http.URLConnectionTest#com.squareup.okhttp.internal.http.URLEncodingTest#com.squareup.okhttp.internal.FaultRecoveringOutputStreamTest#com.squareup.okhttp.apache.OkApacheClientTest#com.squareup.okhttp.internal.spdy.Http20Draft09Test#com.squareup.okhttp.internal.spdy.SpdyConnectionTest#com.squareup.okhttp.internal.spdy.HpackDraft05Test#com.squareup.okhttp.internal.spdy.ByteArrayPoolTest#com.squareup.okhttp.internal.spdy.SettingsTest#com.squareup.okhttp.mockwebserver.CustomDispatcherTest#com.squareup.okhttp.mockwebserver.MockWebServerTest#com.squareup.okhttp.MediaTypeTest#com.squareup.okhttp.ConnectionPoolTest#com.squareup.okhttp.AsyncApiTest#com.squareup.okhttp.RequestTest#com.squareup.okhttp.DispatcherTest#com.squareup.okhttp.internal.tls.HostnameVerifierTest#com.squareup.okhttp.internal.StrictLineReaderTest#com.squareup.okhttp.internal.http.HttpOverSpdy3Test#com.squareup.okhttp.internal.http.HttpResponseCacheTest#com.squareup.okhttp.internal.http.HttpOverHttp20Draft09Test#com.squareup.okhttp.internal.http.StatusLineTest#com.squareup.okhttp.internal.http.RouteSelectorTest#com.squareup.okhttp.internal.http.HeadersTest#com.squareup.okhttp.internal.http.URLConnectionTest#com.squareup.okhttp.internal.http.URLEncodingTest#com.squareup.okhttp.internal.FaultRecoveringOutputStreamTest#com.squareup.okhttp.apache.OkApacheClientTest",
// 	Log_tests_num              []int     `json:"tr_log_tests_num"`        //": "3#35#17#3#5#2#14#10#19#8#5#12#16#1#14#114#14#3#13#3#166#2#7#11#3#35#17#3#5#2#14#10#19#8#5#12#16#1#14#114#14#3#13#3#166#2#7#11",
// 	Log_tests_duration         []float32 `json:"tr_log_tests_duration"`   //": "0.093#2.607#0.031#0.001#0#0.126#7.132#0.178#40.865#0.617#0.006#0.013#0.043#0#4.216#5.537#3.299#0#0.002#0.001#10.347#0#0.001#0.873#0.093#2.607#0.031#0.001#0#0.126#7.132#0.178#40.865#0.617#0.006#0.013#0.043#0#4.216#5.537#3.299#0#0.002#0.001#10.347#0#0.001#0.873",
// 	Log_tests_failed           []string  `json:"tr_log_tests_failed"`     //": "",
// 	Log_tests_failed_num       []string  `json:"tr_log_tests_failed_num"` //": "",
// 	Log_testduration           float64   `json:"tr_log_testduration"`
// 	Log_buildduration          float64   `json:"tr_log_buildduration"`
// }

func ParseTravisTorrent() {
	jsonFile, err := os.Open("data/travistorrent/build_logs/square@okhttp/buildlog-data-travis.json")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened square@okhttp travis build log")
	defer jsonFile.Close()

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		fmt.Println(err)
	}
	var decoded []interface{}
	err = json.Unmarshal(byteValue, &decoded)
	if err != nil {
		fmt.Println(err)
	}

	// write results file
	file, err := os.Create("results/okhttp.csv")
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

	for _, v := range decoded {
		build := v.(map[string]interface{})
		// currTime := j.OldTime + (j.OldTime * (j.ChangePercent / float64(100)))
		// diffTime := currTime - j.OldTime
		commit := build["tr_original_commit"]
		method := build["tr_log_tests"]
		tests_num := build["tr_log_tests_num"]
		tests_duration := build["tr_log_tests_duration"]
		if method != nil && method != "" {
			methods := strings.Split(method.(string), "#")
			nums := strings.Split(tests_num.(string), "#")
			durations := strings.Split(tests_duration.(string), "#")
			for i := 0; i < len(methods); i++ {
				fmt.Println(commit)
				fmt.Println(methods[i])
				fmt.Println(nums[i])
				fmt.Println(durations[i])
				s = []string{commit.(string), methods[i], "", "", durations[i], ""}
				data = append(data, s)
			}
		}
	}
	err = writer.WriteAll(data)
	if err != nil {
		log.Fatal("Cannot write to file", err)
	}
}

/*///////////////
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

////////////////*/

// difference returns the elements in `a` that aren't in `b`.
func slicesDiff(a, b []string) []string {
	fmt.Println(a)
	fmt.Println(b)
	mb := make(map[string]struct{}, len(b))
	for _, x := range b {
		mb[x] = struct{}{}
	}
	var diff []string
	for _, x := range a {
		if _, found := mb[x]; !found {
			diff = append(diff, x)
		}
	}
	return diff
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
		mapDiffTime := make(map[string]float64)
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
				var commit, oldTime, currTime, diffTime string
				if len(record) > 4 {
					commit = record[0]
					oldTime = record[2]
					currTime = record[3]
					diffTime = record[4]
					// changePercent := record[5]
				} else {
					commit = record[0]
					oldTime = record[1]
					currTime = record[2]
				}
				// if oldTime != "" && currTime != "" {

				// old time
				ov, err := strconv.ParseFloat(oldTime, 64)
				if oldTime != "" && err != nil {
					fmt.Println("Cannot parse float value oldTime: ", err)
				}
				mapOldTime[commit] += ov

				// curr time
				cv, err := strconv.ParseFloat(currTime, 64)
				if currTime != "" && err != nil {
					fmt.Println("Cannot parse float value currTime: ", err)
				}
				mapNewTime[commit] += cv

				// diff time
				dv, err := strconv.ParseFloat(diffTime, 64)
				if dv == 0 {
					dv = cv - ov
				}

				if err != nil {
					fmt.Println("Cannot parse float value diffTime: ", err)
				}
				mapDiffTime[commit] += dv
				// } else {
				// 	if diffTime != "" {
				// 		dv, err := strconv.ParseFloat(diffTime, 64)
				// 		if err != nil {
				// 			fmt.Println("Cannot parse float value diffTime: ", err)
				// 		}
				// 		mapDiffTime[commit] += dv
				// 	}
				// }

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
			refDiffTime := fmt.Sprintf("%f", mapDiffTime[k])
			var res = []string{k, resOldTime, resNewTime, refDiffTime}
			err = csvwriter.Write(res)
			if err != nil {
				fmt.Println("Cannot write sum to file: ", err)
			}
		}

		csvwriter.Flush()

		csvfile.Close()
	}
}
