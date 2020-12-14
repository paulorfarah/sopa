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

func ReadSmellsFromCommits(urls map[string]string) {

	//Designite
	smellTool := "organic" //"designite"

	//header columns
	header := "project,commit,order"
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

	header += ", resptime"
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
			commits := make(map[string]string)
			for {
				commit, err := rSumFile.Read()
				if err == io.EOF {
					break
				}
				if err != nil {
					fmt.Println(">>> [ERROR]: Cannot read commit: ", err)
				}
				if commit[0] != "commit" {
					commits[commit[0]] = commit[1]
				}
			}
			// repo := CloneRepo(urls[repoName], repoName)

			// prevCommits := TraverseCommitsWithPrevious(repo, commits)
			// fmt.Println("--- prevCommits ---")
			// fmt.Println(prevCommits)

			// time
			tpath := "results" + string(os.PathSeparator) + "sum" + string(os.PathSeparator) + filename
			times := readTime(tpath)

			for currCommit, prevCommit := range commits {
				fmt.Printf("curr: %s, prev: %s\n", currCommit, prevCommit)
				//curr commit
				processCommit(repoName, currCommit)
				// previous commit
				processCommit(repoName, prevCommit)
				// //summarize results
				var data string
				if smellTool == "designite" {
					data = summarizeDesigniteSmells(repoName, prevCommit, "Previous", designSmells, implSmells)
				} else if smellTool == "organic" {
					data = summarizeOrganicSmells(repoName, prevCommit, "Previous", designSmells, implSmells)
				}

				_, found := times[currCommit]
				if found {
					fmt.Println("found curr: ", currCommit)
					oldTime := fmt.Sprintf("%f", times[currCommit].OldTime)
					data += "," + oldTime
					//write previous commit results
					// _, _ = wSumSmell.WriteString(data + "\n")
					// wSumSmell.Flush()

					// //curr commit
					if smellTool == "designite" {
						data += "," + summarizeDesigniteSmells(repoName, currCommit, "Current", designSmells, implSmells)
					} else if smellTool == "organic" {
						data += "," + summarizeOrganicSmells(repoName, currCommit, "Current", designSmells, implSmells)
					}
					//time
					newTime := fmt.Sprintf("%f", times[currCommit].NewTime)
					data += "," + newTime

					//diff time
					diffTime := fmt.Sprintf("%f", times[currCommit].DiffTime)
					data += "," + diffTime

					//save data file
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

func summarizeDesigniteSmells(repoName, commit, order string, designSmells, implSmells []string) string {
	//summarize results
	pathSmells := "results" + string(os.PathSeparator) + repoName + string(os.PathSeparator) + commit + string(os.PathSeparator) + "smells" + string(os.PathSeparator)
	data := repoName + "," + commit + "," + order

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

func summarizeOrganicSmells(repoName, commit, order string, classSmells, methodSmells []string) string {
	//summarize results
	data := repoName + "," + commit + "," + order
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

func GetParentCommit(repo *git.Repository, hash plumbing.Hash) string {
	var h string
	var prevCommit *object.Commit
	prevCommit = nil
	var prevTree *object.Tree
	prevTree = nil
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
					}
					// prevHash := fmt.Sprintf("%s", prevCommit.Hash)
					// fmt.Printf("hash: %s | h: %s - prev: %s\n", h, prevHash)
					return nil
					// if findCommit(commits, hash) == true {
					// 	prevCommits[hash] = prevHash
					// }
				}
			}
			prevCommit = c
			prevTree, _ = c.Tree()
			return nil
		})
		if err != nil {
			fmt.Println("Error iterating over git log...")
		}
	}
	return h
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

type StrTime struct {
	Commit   string
	OldTime  float64
	NewTime  float64
	DiffTime float64
}

func readTime(path string) map[string]StrTime {
	fmt.Println(path)
	mTime := make(map[string]StrTime)
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
	for i, row := range rows {
		if i != 0 {
			if row != nil {
				fmt.Println(row)
				if len(row) > 2 {
					ot, err := strconv.ParseFloat(row[2], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert OldTime to float", err)
					}
					nt, err := strconv.ParseFloat(row[3], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert NewTime to float", err)
					}
					dt, err := strconv.ParseFloat(row[4], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert NewTime to float", err)
					}

					commit := row[0]
					t := StrTime{Commit: commit, OldTime: ot, NewTime: nt, DiffTime: dt}
					// mTime = append(mTime, t)
					mTime[commit] = t
				}
			}
		}
	}
	return mTime
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
