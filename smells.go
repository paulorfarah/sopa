package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
)

type Element struct {
	Methods            []Method     `json:"methods"`
	SourceFile         FilePath     `json:"source_file"`
	MetricsValues      ClassMetrics `json:"metrics_values"`
	FullyQualifiedName string       `json:"fuly_qualified_name"`
	Smells             []Smell      `json:"smells"`
	Kind               string       `json:"kind"`
}

type Method struct {
	ParametersTypes    []string      `json:"parametersTypes"`
	MetricsValues      MethodMetrics `json:"metricsValues"`
	FullyQualifiedName string        `json:"fullyQualifiedName"`
	Smells             []Smell       `json:"smells"`
}

type MethodMetrics struct {
	ChangingClasses           float32
	CouplingIntensity         float32
	CyclomaticComplexity      float32
	MaxCallChain              float32
	MethodLinesOfCode         float32
	ParameterCount            float32
	CouplingDispersion        float32
	ChangingMethods           float32
	NumberOfAccessedVariables float32
	MaxNesting                float32
}

type Smell struct {
	Name         string
	Reason       string
	StartingLine int
	EndingLine   int
}

type FilePath struct {
	FileRelativePath string
}

type ClassMetrics struct {
	TightClassCohesion      float32
	WeighOfClass            float32
	OverrideRatio           string
	ClassLinesOfCode        float32
	LCOM2                   float32
	FANIN                   float32
	FANOUT                  float32
	WeightedMethodCount     float32
	NumberOfAccessorMethods float32
	LCOM3                   float32
	PublicFieldCount        float32
	IsAbstract              float32
}

type timestamps struct {
	prevCommit    string
	timestamp     time.Time
	prevTimestamp time.Time
}

func ReadSmellsFromCommits(urls map[string]string) {

	//Designite
	smellTool := "organic" //"designite"

	//header columns
	header := "project, timestamp, commit, order"
	var designSmells []string
	var implSmells []string

	if smellTool == "designite" {
		designSmells = []string{"Imperative Abstraction", "Multifaceted Abstraction", "Unnecessary Abstraction", "Unutilized Abstraction",
			"Deficient Encapsulation", "Unexploited Encapsulation", "Broken Modularization", "Cyclic-Dependent Modularization",
			"Insufficient Modularization", "Hub-like Modularization", "Broken Hierarchy", "Cyclic Hierarchy", "Deep Hierarchy",
			"Missing Hierarchy", "Multipath Hierarchy", "Rebellious Hierarchy", "Wide Hierarchy"}
		for _, smell := range designSmells {
			header += "," + smell
		}
		implSmells = []string{"Abstract Function Call From Constructor", "Complex Conditional", "Complex Method", "Empty catch clause",
			"Long Identifier", "Long Method", "Long Parameter List", "Long Statement", "Magic Number", "Missing default"}
		for _, smell := range implSmells {
			header += ", " + smell
		}
	} else {
		designSmells = []string{"GodClass", "ClassDataShouldBePrivate", "ComplexClass", "LazyClass", "RefusedBequest", "SpaghettiCode",
			"SpeculativeGenerality", "DataClass", "BrainClass"}
		for _, smell := range designSmells {
			header += "," + smell
		}
		implSmells = []string{"FeatureEnvy", "LongMethod", "LongParameterList", "MessageChain", "DispersedCoupling", "IntensiveCoupling",
			"ShotgunSurgery", "BrainMethod"}
		for _, smell := range implSmells {
			header += ", " + smell
		}
	}

	header += ", runtime"
	runSmellTool(urls, smellTool, header, designSmells, implSmells)
}

func runSmellTool(urls map[string]string, smellTool, header string, designSmells, implSmells []string) {
	dir := filepath.FromSlash("results/sum/")
	smellsFilename := "sum_" + smellTool + "_smells"
	fSumSmell, err := os.OpenFile("results"+string(os.PathSeparator)+"sum"+string(os.PathSeparator)+smellsFilename+".csv", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed creating fSumSmell: %s", err)
	}
	wSumSmell := bufio.NewWriter(fSumSmell)
	_, _ = wSumSmell.WriteString(header + "\n")
	for repoName := range urls {
		filename := "sum_" + repoName + ".csv"
		fmt.Println("###########################################")
		fmt.Println("# ", repoName)
		fmt.Println("###########################################")
		isCloned := CloneRepo(urls[repoName], repoName)
		if isCloned != nil {
			sumFile, err := os.Open(dir + filename)
			if err != nil {
				log.Fatal("Cannot open file ", err)
			}

			rSumFile := csv.NewReader(sumFile)
			// commits := make(map[string]string)
			// timestamps := make(map[string]time.Time)
			commits := make(map[string]timestamps)
			for {
				commit, err := rSumFile.Read()
				if err == io.EOF {
					break
				}
				if err != nil {
					fmt.Println(">>> [ERROR]: Cannot read commit: ", err)
				}
				if commit[1] != "commit" {
					layout := "2006-01-02 15:04:05 -0700 -0700"
					str := commit[0]
					t, _ := time.Parse(layout, str)
					strp := commit[2]
					pt, _ := time.Parse(layout, strp)
					var ts timestamps
					ts.timestamp = t
					ts.prevTimestamp = pt
					ts.prevCommit = commit[3]
					commits[commit[1]] = ts
					// timestamps[commit[1]] = t
					// commits[commit[1]] = commit[3]
				}
			}

			// time
			tpath := "results" + string(os.PathSeparator) + "sum" + string(os.PathSeparator) + filename
			metrics := readMetrics(tpath) //mMetric

			for currCommit, ts := range commits {
				fmt.Printf("prev: %s %s, curr:%s %s\n", ts.prevTimestamp, ts.prevCommit, ts.timestamp, currCommit)
				//curr commit
				processCommit(repoName, currCommit)
				// previous commit
				processCommit(repoName, ts.prevCommit)
				// //summarize results
				data := repoName + "," + ts.prevCommit + "," + ts.prevTimestamp.String() + "," + "Previous"
				if smellTool == "designite" {
					data += summarizeDesigniteSmells(repoName, ts.prevCommit, designSmells, implSmells)
				} else if smellTool == "organic" {
					data += summarizeOrganicSmells(repoName, ts.prevCommit, designSmells, implSmells)
				}

				_, found := metrics[currCommit]
				fmt.Println(currCommit, found)
				if found {
					// fmt.Println("found curr: ", currCommit)
					oldTime := fmt.Sprintf("%f", metrics[currCommit].oldTime)
					data += currCommit + ", " + ts.timestamp.String() + ", " + "Current, " + oldTime
					
					// //curr commit
					if smellTool == "designite" {
						data += "," + summarizeDesigniteSmells(repoName, currCommit, designSmells, implSmells)
					} else if smellTool == "organic" {
						data += "," + summarizeOrganicSmells(repoName, currCommit, designSmells, implSmells)
					}
					//time
					newTime := fmt.Sprintf("%f", metrics[currCommit].newTime)
					data += "," + newTime

					//diff time
					diffTime := fmt.Sprintf("%f", metrics[currCommit].diffTime)
					data += "," + diffTime

					//save data file
					fmt.Println("save data: ", data)
					_, _ = wSumSmell.WriteString(data + "\n")
					wSumSmell.Flush()
				}
			}
		} else {
			fmt.Println("Cannot clone repository: ", repoName)
		}
	}
	fSumSmell.Close()
}

func processCommit(repoName, commit string) {
	fmt.Printf("git --git-dir=repos"+string(os.PathSeparator)+"%v"+string(os.PathSeparator)+".git --work-tree=repos"+string(os.PathSeparator)+"%v checkout %s\n", repoName, repoName, commit)
	_, err := exec.Command("git", "--git-dir=repos"+string(os.PathSeparator)+repoName+string(os.PathSeparator)+".git", "--work-tree=repos"+string(os.PathSeparator)+repoName, "checkout", commit).Output()
	if err != nil {
		fmt.Println("\nCannot run git checkout: ", err)
	}
	// ProcessMetrics(repoName, commit)
	ProcessSmells(repoName, commit)
}

func summarizeDesigniteSmells(repoName, commit string, designSmells, implSmells []string) string {
	//summarize results
	pathSmells := "results" + string(os.PathSeparator) + repoName + string(os.PathSeparator) + commit + string(os.PathSeparator) + "smells" + string(os.PathSeparator)
	data := ", " //repoName + "," + commit + "," + order

	//design smells
	sumDSmells := readSmellsCsv(pathSmells+"DesignSmells.csv", 3)
	for _, smell := range designSmells {
		data += "," + strconv.Itoa(sumDSmells[smell])
	}

	//implementation smells
	sumISmells := readSmellsCsv(pathSmells+"ImplementationSmells.csv", 4)
	for _, smell := range implSmells {
		data += "," + strconv.Itoa(sumISmells[smell])
	}
	return data
}

func summarizeOrganicSmells(repoName, commit string, classSmells, methodSmells []string) string {
	//summarize results
	data := "" //repoName + "," + commit + "," + order
	pathSmells := "results" + string(os.PathSeparator) + repoName + string(os.PathSeparator) + commit + string(os.PathSeparator) + "smells" + string(os.PathSeparator)
	jsonFile, err := os.Open(pathSmells + "smells_organic.json")
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var result []Element //map[string]interface{}
	json.Unmarshal([]byte(byteValue), &result)

	smellQt := make(map[string]int)
	for _, element := range result {
		// fmt.Println("Element Smells: ", element.Smells)
		for _, cs := range element.Smells {
			smellQt[cs.Name]++
			// fmt.Println(cs.Name)
		}
		// fmt.Println(">>>Metods: ", element.Methods)
		for _, m := range element.Methods {
			// fmt.Println("Method Smells: ", m.Smells)
			for _, ms := range m.Smells {
				smellQt[ms.Name]++
				// fmt.Println(ms.Name)
			}
		}
	}

	// fmt.Println("*** class Smells: ", classSmells)
	for _, smell := range classSmells {
		data += "," + strconv.Itoa(smellQt[smell])
	}

	// fmt.Println("*** method Smells: ", methodSmells)
	for _, smell := range methodSmells {
		data += "," + strconv.Itoa(smellQt[smell])
	}
	fmt.Println("data: ", data)
	return data
}

func CloneRepo(url, directory string) *git.Repository {
	path := "repos" + string(os.PathSeparator) + directory
	os.RemoveAll(path)
	fmt.Printf("git clone -n %v repos"+string(os.PathSeparator)+"%v\n", url, directory)
	r, err := git.PlainClone(path, false, &git.CloneOptions{
		URL: url,
	})
	if err != nil {
		fmt.Println("Error cloning repository: ", err)
	}
	return r
}

func OpenRepository(directory string) *git.Repository {
	path := "repos" + string(os.PathSeparator) + directory
	r, err := git.PlainOpen(path)
	if err != nil {
		fmt.Println("Cannot open repository: ", err)
	}
	return r
}

func GetParentsFromCommit(repo *git.Repository, hash string) []string {
	var parentHashes []string
	h := plumbing.NewHash(hash)
	commit, err := repo.CommitObject(h)
	if err != nil {
		fmt.Println("Cannot read Head commit: ", err)
	}

	// retrieves the commit history
	parents := commit.ParentHashes

	// iterates over the commits and print each
	for _, p := range parents {
		parentHashes = append(parentHashes, p.String())
	}
	return parentHashes
}

func GetParentCommit(repo *git.Repository, hash plumbing.Hash) (string, time.Time, time.Time) {
	var h string
	var prevCommit *object.Commit
	prevCommit = nil
	var prevTree *object.Tree
	prevTree = nil
	var commiterWhen, prevCommiterWhen time.Time

	if repo != nil {
		cIter, err := repo.Log(&git.LogOptions{From: hash})
		if err != nil {
			fmt.Println("Cannot get log history of repository")
		}

		err = cIter.ForEach(func(c *object.Commit) error {
			// fmt.Printf("%s\n", c.Hash)
			if prevCommit != nil {
				if prevTree != nil {
					if h == "" {
						// h = fmt.Sprintf("%s", c.Hash)
						h = c.Hash.String()
						prevCommiterWhen = c.Author.When
					}
					// prevHash := fmt.Sprintf("%s", prevCommit.Hash)
					// fmt.Printf("hash: %s | h: %s - prev: %s\n", h, prevHash)
					return nil
					// if findCommit(commits, hash) == true {
					// 	prevCommits[hash] = prevHash
					// }
				}
			} else {
				commiterWhen = c.Author.When
			}
			prevCommit = c
			prevTree, _ = c.Tree()
			return nil
		})
		if err != nil {
			fmt.Println("Error iterating over git log...")
		}
	}
	return h, commiterWhen, prevCommiterWhen
}

func TraverseCommitsWithPrevious(repo *git.Repository, commits []string) map[string]string {
	var prevCommits = make(map[string]string)
	if repo != nil {
		ref, err := repo.Head()
		if err != nil {
			//log.Fatal(err)
			fmt.Println("[ERROR]>> Cannot get Head commit of repository ", err)
		}
		if ref != nil {
			var prevCommit *object.Commit
			prevCommit = nil
			var prevTree *object.Tree
			prevTree = nil

			// fmt.Println(ref.Hash())
			cIter, err := repo.Log(&git.LogOptions{From: ref.Hash()})
			if err != nil {
				//log.Fatal(err)
				fmt.Println("Cannot get log history of repository ", err)
			}
			err = cIter.ForEach(func(c *object.Commit) error {
				if prevCommit != nil {
					if prevTree != nil {
						hash := fmt.Sprintf("%s", c.Hash)
						prevHash := fmt.Sprintf("%s", prevCommit.Hash)
						// fmt.Printf("curr: %s - prev: %s\n", hash, prevHash)
						if findCommit(commits, hash) == true {
							prevCommits[hash] = prevHash
						}
					}
				}
				prevCommit = c
				prevTree, _ = c.Tree()
				return nil
			})
			if err != nil {
				fmt.Println(">>> [ERROR]: Cannot iterate over commits", err)
			}
		} else {
			fmt.Println("Cannot set HEAD reference...")
		}
	} else {
		fmt.Println("[ERROR]>> repository is nil.")
	}
	fmt.Println("############################# PREVCOMMITS BEGIN")
	fmt.Println(prevCommits)
	fmt.Println("############################# PREVCOMMITS END")
	return prevCommits
}

func findCommit(commits []string, commit string) bool {
	for _, n := range commits {
		if commit == n {
			return true
		}
	}
	return false
}

func ProcessMetrics(repo string, commit string) {
	path := "results" + string(os.PathSeparator) + repo + string(os.PathSeparator) + commit + string(os.PathSeparator) + "metrics"
	checkDirectory(path)
	createUnderstandDb(repo, path)
	runUnderstand(path)
}

func createUnderstandDb(repo, path string) {
	_, err := exec.Command("und", "create", "-db", path+string(os.PathSeparator)+"understand.udb", "-languages", "java", "add", "repos"+string(os.PathSeparator)+repo).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Cannot create understand database: ", err)
	}
}

func runUnderstand(path string) {
	os.Remove("und.txt")
	cmd := path + string(os.PathSeparator) + `understand.udb
settings -MetricMetrics "AvgCyclomatic" "AvgCyclomaticModified" "AvgCyclomaticStrict" "AvgEssential" "AvgLine" "AvgLineBlank" "AvgLineCode" "AvgLineComment" "CountClassBase" "CountClassCoupled" "CountClassCoupledModified" "CountClassDerived" "CountDeclClass" "CountDeclClassMethod" "CountDeclClassVariable" "CountDeclExecutableUnit" "CountDeclFile" "CountDeclFunction" "CountDeclInstanceMethod" "CountDeclInstanceVariable" "CountDeclMethod" "CountDeclMethodAll" "CountDeclMethodDefault" "CountDeclMethodPrivate" "CountDeclMethodProtected" "CountDeclMethodPublic" "CountInput" "CountLine" "CountLineBlank" "CountLineCode" "CountLineCodeDecl" "CountLineCodeExe" "CountLineComment" "CountOutput" "CountPath" "CountPathLog" "CountSemicolon" "CountStmt" "CountStmtDecl" "CountStmtExe" "Cyclomatic" "CyclomaticModified" "CyclomaticStrict" "Essential" "Knots" "MaxCyclomatic" "MaxCyclomaticModified" "MaxCyclomaticStrict" "MaxEssential" "MaxEssentialKnots" "MaxInheritanceTree" "MaxNesting" "MinEssentialKnots" "PercentLackOfCohesion" "PercentLackOfCohesionModified" "RatioCommentToCode" "SumCyclomatic" "SumCyclomaticModified" "SumCyclomaticStrict" "SumEssential"
analyze
metrics
`
	d := []byte(cmd)
	err := ioutil.WriteFile("und.txt", d, 0644)
	if err != nil {
		fmt.Println("[ERROR]>> Cannot create understand und.txt file", err)
	}

	//fmt.Printf("und process und.txt\n")
	_, err = exec.Command("und", "process", "und.txt").Output()

	if err != nil {
		fmt.Println("[ERROR]>> Cannot add repository path: ", err)
	}
}

func ProcessSmells(repo, commit string) {
	path := "results" + string(os.PathSeparator) + repo + string(os.PathSeparator) + commit + string(os.PathSeparator) + "smells"
	checkDirectory(path)
	// runDesignite(repo, path)
	runOrganic(repo, path)
}

func runDesignite(repo, path string) {
	_, err := exec.Command("java", "-jar", "DesigniteJava.jar", "-i", "repos"+string(os.PathSeparator)+repo, "-o", path).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Error trying to generate Designite smells files: ", err)
	}
}

func runOrganic(repo, path string) {
	// java -jar -XX:MaxPermSize=2560m -Xms40m -Xmx2500m ${EQUINOX} -application ${ORGANIC} -sf "organic_smells.json" -src
	// "/mnt/sda4/farah/go-work/src/github.com/paulorfarah/sopa/repos/hadoop/"
	fmt.Println("/mnt/sda4/organic/bin/organic", "-sf", path+"/smells_organic.json", "-src", "repos"+string(os.PathSeparator)+repo)
	_, err := exec.Command("/mnt/sda4/organic/bin/organic", "-sf", path+"/smells_organic.json", "-src", "repos"+string(os.PathSeparator)+repo).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Error trying to generate organic smells files: ", err)
	}
}

func checkDirectory(path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		dirs := strings.Split(path, string(os.PathSeparator))
		subpath := ""
		for _, dir := range dirs {
			os.Mkdir(subpath+dir, 0755)
			subpath += dir + string(os.PathSeparator)
		}
	}
}

func readSmellsCsv(path string, column int) map[string]int {
	// fmt.Println(path)
	sumSmells := make(map[string]int)

	f, err := os.Open(path)
	if err != nil {
		fmt.Println("Cannot open smell file", err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.Comma = ','
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		fmt.Printf("Cannot read csv data 1: %s", path)
		fmt.Println(err)
	}
	for i, row := range rows {
		if i != 0 {
			if row != nil {
				if len(row) > column {
					sumSmells[row[column]]++
				}
			}
		}
	}
	return sumSmells
}

type strMetrics struct {
	commit     string
	oldTime    float64
	newTime    float64
	diffTime   float64
	oldCpu     float64
	newCpu     float64
	diffCpu    float64
	oldMemory  float64
	newMemory  float64
	diffMemory float64
	oldIo      float64
	newIo      float64
	diffIo     float64
}

func readMetrics(path string) map[string]strMetrics {
	mMetric := make(map[string]strMetrics)
	f, err := os.Open(path)
	if err != nil {
		fmt.Println("Cannot open time file", err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.Comma = ','
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		fmt.Println("Cannot read csv data of time file", err)
	}
	for _, row := range rows {
		fmt.Println("time row: ", row)
		if row != nil {
			// fmt.Println(row)
			if len(row) > 2 {
				commit := row[0]
				//runtime
				ot, err := strconv.ParseFloat(row[4], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert runtime OldMetric to float", err)
				}
				nt, err := strconv.ParseFloat(row[5], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert runtime NewMetric to float", err)
				}
				dt, err := strconv.ParseFloat(row[6], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert runtime DiffMetric to float", err)
				}

				//cpu
				oc := float64(0.0)
				nc := float64(0.0)
				dc := float64(0.0)
				if row[10] != "" && row[11] != "" {
				oc, err = strconv.ParseFloat(row[7], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert cpu OldMetric to float", err)
				}
				nc, err = strconv.ParseFloat(row[8], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert cpu NewMetric to float", err)
				}
				dc, err = strconv.ParseFloat(row[9], 32)
				if err != nil {
					fmt.Println("### ERROR: Cannot convert cpu DiffMetric to float", err)
				}

				//mem
				om := float64(0.0)
				nm := float64(0.0)
				dm := float64(0.0)
				if row[10] != "" && row[11] != "" {
					om, err = strconv.ParseFloat(row[10], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert memory OldMetric to float", err)
					}
					nm, err = strconv.ParseFloat(row[11], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert memory NewMetric to float", err)
					}
					dm, err = strconv.ParseFloat(row[12], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert memory DiffMetric to float", err)
					}
				}

				//io
				oi := float64(0.0)
				ni := float64(0.0)
				di := float64(0.0)
				if row[13] != "" && row[14] != "" {
					oi, err = strconv.ParseFloat(row[13], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert IO OldMetric to float", err)
					}
					ni, err = strconv.ParseFloat(row[14], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert IO NewMetric to float", err)
					}
					di, err = strconv.ParseFloat(row[15], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert IO DiffMetric to float", err)
					}
				}
				mMetric[commit] = strMetrics{commit: commit, oldTime: ot, newTime: nt, diffTime: dt,
					oldCpu: oc, newCpu: nc, diffCpu: dc,
					oldMemory: om, newMemory: nm, diffMemory: dm,
					oldIo: oi, newIo: ni, diffIo: di}
			}
		}
		//}
	}
	fmt.Println("mMetric: ", mMetric)
	return mMetric
}

//func main() {
//	url := "http://github.com/paulorfarah/refactoring-python-code"
//	dir := "repos"
//	commits := []string{"3599f5cdc72e5526de48df98c212322a835869cd", "ee950eac81d01d30b473d0a0aa74d7a94e22a8e6", "a74d09e3824c152fa0348a9f36e5b2c8af27a181"}
//	res := GetPreviousCommits(url, dir, commits)
//	for k, v := range res {
//		fmt.Printf("%v: %v\n", k, v)
//	}
//}
