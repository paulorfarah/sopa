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
	"time"

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

type commitPerf struct {
	commitTime     time.Time
	commit         string
	prevCommitTime time.Time
	prevCommit     string
	runtime        string
	prevRuntime    string
	diffRuntime    float64
	cpu            string
	prevCpu        string
	diffCpu        float64
	memory         string
	prevMemory     string
	diffMemory     float64
	io             string
	prevIo         string
	diffIo         float64
}

func ParsePeassResults() {
	urlsPeass := map[string]string{
		"commons-compress":   "https://github.com/apache/commons-compress",
		"commons-csv":        "https://github.com/apache/commons-csv",
		"commons-dbcp":       "https://github.com/apache/commons-dbcp",
		"commons-fileupload": "https://github.com/apache/commons-fileupload",
		"commons-imaging":    "https://github.com/apache/commons-imaging",
		"commons-io":         "https://github.com/apache/commons-io",
		"commons-pool":       "https://github.com/apache/commons-pool",
		"commons-text":       "https://github.com/apache/commons-text",
		"jackson-core":       "https://github.com/FasterXML/jackson-core",
		"k-9":                "https://github.com/k9mail/k-9",
	}
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

		url := urlsPeass[resFilename]
		repo := CloneRepo(url, resFilename)

		var result PeassResult
		var data [][]string
		s := []string{"commit", "method", "prevCommit", "oldTime", "currTime", "diffTime", "changePercent"}
		data = append(data, s)
		json.Unmarshal([]byte(byteValue), &result)
		for commit, v := range result.VersionChanges {
			prevCommit, commitTime, prevCommitTime := GetParentCommit(repo, plumbing.NewHash(commit))
			for _, w := range v.TestCaseChanges {
				for _, j := range w {
					currTime := j.OldTime + (j.OldTime * (j.ChangePercent / float64(100)))
					diffTime := currTime - j.OldTime
					s = []string{commitTime.String(), commit, j.Method, prevCommitTime.String(), prevCommit, fmt.Sprintf("%f", j.OldTime), fmt.Sprintf("%f", currTime), fmt.Sprintf("%f", diffTime), fmt.Sprintf("%f", j.ChangePercent)}
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

func ParseTravisTorrent() {
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
	s := []string{"commit", "method", "prevCommit", "oldTime", "currTime", "diffTime", "changePercent"}
	data = append(data, s)
	json.Unmarshal([]byte(byteValue), &result)

	dir := "okhttp"
	url := "https://github.com/square/okhttp"
	repo := CloneRepo(url, dir)

	for _, v := range decoded {
		build := v.(map[string]interface{})
		// currTime := j.OldTime + (j.OldTime * (j.ChangePercent / float64(100)))
		// diffTime := currTime - j.OldTime
		commit := build["tr_original_commit"]
		method := build["tr_log_tests"]
		prevCommit, commitTime, prevCommitTime := GetParentCommit(repo, plumbing.NewHash(fmt.Sprintf("%v", commit)))
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
				s = []string{commitTime.String(), commit.(string), methods[i], prevCommitTime.String(), prevCommit, "", "", durations[i], ""}
				data = append(data, s)
			}
		}
	}
	err = writer.WriteAll(data)
	if err != nil {
		log.Fatal("Cannot write to file", err)
	}
}

func ParseHadoopResults() {
	/*
		read dataset from data folder
	*/
	// dir := "refactoring-python-code"
	// url := "https://github.com/paulorfarah/refactoring-python-code"
	// outfile, errOut := os.Create("results/rpc.csv")
	// infile := "data/hadoop/rpc.csv"

	dir := "hadoop"
	url := "https://github.com/apache/hadoop"
	outfile, errOut := os.Create("results/hadoop.csv")

	repo := CloneRepo(url, dir)

	if errOut != nil {
		log.Fatal("Cannot create hadoop summary results file", errOut)
	}
	defer outfile.Close()
	writer := csv.NewWriter(outfile)
	defer writer.Flush()

	hadoopCommits := make(map[string]*commitPerf)

	//commmit, prevCommit, runtime
	metrics := []string{"runtime"} //, "cpu", "memory", "io"}

	for _, metric := range metrics {
		infile := "data/hadoop/hadoop_" + metric + ".csv"
		inRes := readHadoopCsv(infile)
		for commit, mapMethod := range inRes {
			// fmt.Println(commit, mapMethod)
			prevCommit, commitTime, prevCommitTime := GetParentCommit(repo, plumbing.NewHash(commit))
			mapMethodPrev := inRes[prevCommit]
			mValue, prevValue := sumMetricRow(commit, prevCommit, mapMethod, mapMethodPrev)
			// fmt.Println(commit, mValue)
			// fmt.Println(prevCommit, prevValue)
			// row := commitPerf{commit: commit, prevCommit: prevCommit, runtime: runtime, prevRuntime: prevRuntime}
			switch metric {
			case "runtime":
				hadoopCommits[commit] = &commitPerf{commitTime: commitTime, commit: commit, prevCommitTime: prevCommitTime, prevCommit: prevCommit, runtime: mValue, prevRuntime: prevValue}
				v, _ := strconv.ParseFloat(mValue, 64)
				p, _ := strconv.ParseFloat(prevValue, 64)
				(*hadoopCommits[commit]).diffRuntime = v - p
			case "cpu":
				(*hadoopCommits[commit]).cpu = mValue
				(*hadoopCommits[commit]).prevCpu = prevValue
				v, _ := strconv.ParseFloat(mValue, 64)
				p, _ := strconv.ParseFloat(prevValue, 64)
				(*hadoopCommits[commit]).diffCpu = v - p
			case "memory":
				(*hadoopCommits[commit]).memory = mValue
				(*hadoopCommits[commit]).prevMemory = prevValue
				v, _ := strconv.ParseFloat(mValue, 64)
				p, _ := strconv.ParseFloat(prevValue, 64)
				(*hadoopCommits[commit]).diffMemory = v - p
			case "io":
				(*hadoopCommits[commit]).io = mValue
				(*hadoopCommits[commit]).prevIo = prevValue
				v, _ := strconv.ParseFloat(mValue, 64)
				p, _ := strconv.ParseFloat(prevValue, 64)
				(*hadoopCommits[commit]).diffIo = v - p
			}

		}
	}

	// // cpu
	// f = "data/hadoop/hadoop_cpu.csv"
	// res = readHadoopCsv(f)
	// for commit, mapMethod := range res {
	// 	prevCommit := GetParentCommit(repo, plumbing.NewHash(commit))
	// 	mapMethodPrev := res[prevCommit]
	// 	runtime, prevRuntime := sumMetricRow(commit, prevCommit, mapMethod, mapMethodPrev)
	// 	row := commitPerf{commit: commit, prevCommit: prevCommit, runtime: runtime, prevRuntime: prevRuntime}
	// 	hadoopCommits = append(hadoopCommits, row)
	// }

	for _, c := range hadoopCommits {
		row := []string{c.commitTime.String(), c.commit, "method name", c.prevCommitTime.String(), c.prevCommit, c.prevRuntime, 
		    c.runtime, fmt.Sprintf("%f", c.diffRuntime),
			c.prevCpu, c.cpu, fmt.Sprintf("%f", c.diffCpu),
			c.prevMemory, c.memory, fmt.Sprintf("%f", c.diffMemory),
			c.prevIo, c.io, fmt.Sprintf("%f", c.diffIo)}

		if c.prevRuntime != "" &&
			c.runtime != "" &&
			// c.diffRuntime != float64(0) &&
			c.prevCpu != "" &&
			c.cpu != "" &&
			// c.diffCpu != float64(0) &&
			c.prevMemory != "" &&
			c.memory != "" &&
			// c.diffMemory != float64(0) &&
			c.prevIo != "" &&
			c.io != "" &&
			// c.diffIo != float64(0) {
			writer.Write(row)
		}

	}
}

func readHadoopCsv(f string) map[string]map[string]float64 {

	infile, errIn := os.Open(f)

	var res = make(map[string]map[string]float64)
	if errIn != nil {
		fmt.Println(errIn)
	}
	defer infile.Close()

	lines, err := csv.NewReader(infile).ReadAll()
	if err != nil {
		fmt.Println(err)
	}

	// var commits []string
	for _, line := range lines {
		commit := line[0]
		method := line[1]
		_, ok := res[commit]
		if !ok {
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
	}
	return res
}

func sumMetricRow(commit, prevCommit string, mapMethodCur, mapMethodPrev map[string]float64) (string, string) {
	var (
		sumStr     string
		sumPrevStr string
	)

	// sumMetric := make(map[string]float64)
	// current commit
	sum := float64(0)
	methods := []string{}
	for methodName, methodMetric := range mapMethodCur {
		sum += methodMetric
		methods = append(methods, methodName)
	}
	// sumMetric[commit] = sum

	//previous commit
	// prevCommit := GetParentCommit(repo, plumbing.NewHash(commit))
	sumPrev := float64(0)
	methodsPrev := []string{}
	for methodName, methodMetric := range mapMethodPrev {
		sumPrev += methodMetric
		methodsPrev = append(methodsPrev, methodName)
	}
	methodsDiff := slicesDiff(methods, methodsPrev)
	// fmt.Println("methods: ", methods)
	// fmt.Println("methodsPrev: ", methodsPrev)
	// fmt.Println("methodsDiff: ", methodsDiff)
	if len(methodsDiff) == 0 {
		sumStr = fmt.Sprintf("%f", sum)
		sumPrevStr = fmt.Sprintf("%f", sumPrev)
	}
	return sumStr, sumPrevStr
}

// difference returns the elements in `a` that aren't in `b`.
// func slicesDiff(a, b []string) []string {
// 	mb := make(map[string]struct{}, len(b))
// 	for _, x := range b {
// 		mb[x] = struct{}{}
// 	}
// 	var diff []string
// 	for _, x := range a {
// 		if _, found := mb[x]; !found {
// 			diff = append(diff, x)
// 		}
// 	}
// 	return diff
// }

func slicesDiff(slice1 []string, slice2 []string) []string {
	var diff []string

	// Loop two times, first to find slice1 strings not in slice2,
	// second loop to find slice2 strings not in slice1
	for i := 0; i < 2; i++ {
		for _, s1 := range slice1 {
			found := false
			for _, s2 := range slice2 {
				if s1 == s2 {
					found = true
					break
				}
			}
			// String not found. We add it to return slice
			if !found {
				diff = append(diff, s1)
			}
		}
		// Swap the slices, only if it was the first loop
		if i == 0 {
			slice1, slice2 = slice2, slice1
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

// type Result struct {
// }

func SummarizeResults() {
	dir := filepath.FromSlash("results/")
	abs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal("Cannot read absolute filepath", err)
	}
	files := getFiles(abs, ".csv")

	for _, filename := range files {
		// mapPrevCommit := make(map[string]string)
		// mapOldTime := make(map[string]float64)
		// mapNewTime := make(map[string]float64)
		// mapDiffTime := make(map[string]float64)
		mapCommitPerf := make(map[string]*commitPerf)
		csvfile, err := os.Open("results/" + filename)
		if err != nil {
			log.Fatal("Cannot open file ", err)
		}

		// Parse the file
		r := csv.NewReader(csvfile)
		// Iterate through the records
		for {
			// Read each record from csv
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				fmt.Println("Cannot read row: ", filename, err)
			}
			layout := "2006-01-02 15:04:05 -0700 -0700" //"2006-01-02T15:04:05.000Z"
			//"2017-06-28 10:50:09 +0900 +0900"
			t, err := time.Parse(layout, record[0])
			if err != nil {
				fmt.Println("Cannot parse currTime", err)
			}
			commit := strings.TrimSpace(record[1])

			if commit != "commit" {
				fmt.Printf("row: %s\n", record)
				//commit,method,oldTime,currTime,diffTime,changePercent
				// var prevCommit, oldTime, currTime, diffTime string

				prevTime, err := time.Parse(layout, record[3])
				if err != nil {
					fmt.Println("Cannot parse prevTime", err)
				}

				dr, _ := strconv.ParseFloat(record[5], 64)

				mapCommitPerf[commit] = &commitPerf{
					commitTime:     t,
					prevCommitTime: prevTime,
					prevCommit:     record[4],
					prevRuntime:    record[5],
					runtime:        record[6],
					diffRuntime:    dr,
				}
				cols := len(record)
				if cols >= 15 {
					//cpu
					mapCommitPerf[commit].prevCpu = record[7]
					mapCommitPerf[commit].cpu = record[8]
					dc, _ := strconv.ParseFloat(record[9], 64)
					mapCommitPerf[commit].diffCpu = dc
					//memory
					mapCommitPerf[commit].prevMemory = record[10]
					mapCommitPerf[commit].memory = record[11]
					dm, _ := strconv.ParseFloat(record[12], 64)
					mapCommitPerf[commit].diffMemory = dm
					//io
					mapCommitPerf[commit].prevIo = record[13]
					mapCommitPerf[commit].io = record[14]
					di, _ := strconv.ParseFloat(record[15], 64)
					mapCommitPerf[commit].diffIo = di
				} else {
					// changePercent := record[5]
					mapCommitPerf[commit].prevCommit = record[2]
					mapCommitPerf[commit].prevRuntime = record[3]
					mapCommitPerf[commit].runtime = record[4]
				}

				// // mapPrevCommit[commit] = prevCommit

				// // old time
				// ov, err := strconv.ParseFloat(oldTime, 64)
				// if oldTime != "" && err != nil {
				// 	fmt.Println("Cannot parse float value oldTime: ", err)
				// }
				// mapOldTime[commit] += ov

				// // curr time
				// cv, err := strconv.ParseFloat(currTime, 64)
				// if currTime != "" && err != nil {
				// 	fmt.Println("Cannot parse float value currTime: ", err)
				// }
				// mapNewTime[commit] += cv

				// // diff time
				// var dv float64
				// if diffTime == "" {
				// 	dv = cv - ov
				// } else {
				// 	dv, err = strconv.ParseFloat(diffTime, 64)
				// 	if dv == 0 {
				// 		dv = cv - ov
				// 	}

				// 	if err != nil {
				// 		fmt.Println("Cannot parse float value diffTime: ", err)
				// 	}
				// }
				// mapDiffTime[commit] += dv

				_, exists := mapCommitPerf[commit]
				if exists {
					if mapCommitPerf[commit].diffRuntime == 0 {
						r, _ := strconv.ParseFloat(mapCommitPerf[commit].runtime, 64)
						pr, _ := strconv.ParseFloat(mapCommitPerf[commit].prevRuntime, 64)
						mapCommitPerf[commit].diffRuntime = r - pr
					}
				}
			}
		}

		//save summary file
		outfile, err := os.Create("results/sum/sum_" + filename)

		if err != nil {
			log.Fatalf("failed creating file: %s", err)
		}

		csvwriter := csv.NewWriter(outfile)

		// for k, v := range mapOldTime {
		// 	resOldTime := fmt.Sprintf("%f", v)
		// 	resNewTime := fmt.Sprintf("%f", mapNewTime[k])
		// 	refDiffTime := fmt.Sprintf("%f", mapDiffTime[k])
		// 	var res = []string{k, mapPrevCommit[k], resOldTime, resNewTime, refDiffTime}
		// 	err = csvwriter.Write(res)
		// 	if err != nil {
		// 		fmt.Println("Cannot write sum to file: ", err)
		// 	}
		// }

		// fmt.Println("mapCommitPerf...")
		// fmt.Println(mapCommitPerf)

		for k, v := range mapCommitPerf {
			fmt.Println(k, v)
			var res = []string{v.commitTime.String(), k, v.prevCommitTime.String(), v.prevCommit, v.prevRuntime, v.runtime, fmt.Sprintf("%f", v.diffRuntime),
				v.prevCpu, v.cpu, fmt.Sprintf("%f", v.diffCpu),
				v.prevMemory, v.memory, fmt.Sprintf("%f", v.diffMemory),
				v.prevIo, v.io, fmt.Sprintf("%f", v.diffIo)}
			err = csvwriter.Write(res)
			if err != nil {
				fmt.Println("Cannot write sum to file: ", err)
			}
		}
		csvwriter.Flush()
		csvfile.Close()
	}
}
