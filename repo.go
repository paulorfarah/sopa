package main

import (
	"bufio"
	"encoding/csv"
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
	"github.com/go-git/go-git/v5/plumbing/object"
)

func ReadCommits() {

	urls := map[string]string{
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

	dir := filepath.FromSlash("results/sum/")
	//open file
	// abs, err := filepath.Abs(dir)
	// if err != nil {
	// 	log.Fatal("Cannot read absolute filepath", err)
	// }
	// files := getFiles(abs, ".csv")

	file, err := os.OpenFile("results\\sum\\sumsmells.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	datawriter := bufio.NewWriter(file)

	//header columns
	header := "project,commit,order"
	designSmells := []string{"Imperative Abstraction", "Multifaceted Abstraction", "Unnecessary Abstraction", "Unutilized Abstraction", "Deficient Encapsulation", "Unexploited Encapsulation", "Broken Modularization", "Cyclic-Dependent Modularization", "Insufficient Modularization", "Hub-like Modularization", "Broken Hierarchy", "Cyclic Hierarchy", "Deep Hierarchy", "Missing Hierarchy", "Multipath Hierarchy", "Rebellious Hierarchy", "Wide Hierarchy"}
	for _, smell := range designSmells {
		header += "," + smell
	}
	implSmells := []string{"Abstract Function Call From Constructor", "Complex Conditional", "Complex Method", "Empty catch clause", "Long Identifier", "Long Method", "Long Parameter List", "Long Statement", "Magic Number", "Missing default"}
	for _, smell := range implSmells {
		header += ", " + smell
	}
	header += ", resptime"
	_, _ = datawriter.WriteString(header + "\n")

	for repoName := range urls {
		// repoName := strings.ReplaceAll(filename, ".csv", "")
		// repoName = strings.ReplaceAll(repoName, "sum_peass_", "")
		filename := "sum_" + repoName + ".csv"
		fmt.Println("###########################################")
		fmt.Println("# ", repoName)
		fmt.Println("###########################################")
		isCloned := CloneRepo(urls[repoName], repoName)
		if isCloned == true {
			sumFile, err := os.Open(dir + filename)
			if err != nil {
				log.Fatal("Cannot open file ", err)
			}

			r := csv.NewReader(sumFile)
			var commits []string
			for {
				commit, err := r.Read()
				if err == io.EOF {
					break
				}
				if err != nil {
					fmt.Println(">>> [ERROR]: Cannot read commit: ", err)
				}
				if commit[0] != "commit" {
					commits = append(commits, commit[0])
				}
			}
			prevCommits := GetPreviousCommits(urls[repoName], repoName, commits)

			// time
			times := readTime("results\\sum\\" + filename)

			for currCommit, prevCommit := range prevCommits {
				//fmt.Printf("curr: %s, prev: %s\n", currCommit, prevCommit)
				//curr commit
				// fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repoName, repoName, currCommit)
				// _, err := exec.Command("git", "--git-dir=repos\\"+repoName+"\\.git", "--work-tree=repos\\"+repoName, "checkout", currCommit).Output()
				// if err != nil {
				// 	fmt.Println("\nCannot run git checkout: ", err)
				// }
				// ProcessMetrics(repoName, currCommit)
				// ProcessSmells(repoName, currCommit)
				processCommit(repoName, currCommit)

				// previous commit
				// fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repoName, repoName, prevCommit)
				// _, err = exec.Command("git", "--git-dir=repos\\"+repoName+"\\.git", "--work-tree=repos\\"+repoName, "checkout", prevCommit).Output()
				// if err != nil {
				// 	fmt.Println("\nCannot run git checkout: ", err)
				// }
				// ProcessMetrics(repoName, prevCommit)
				// ProcessSmells(repoName, prevCommit)
				processCommit(repoName, prevCommit)

				// //summarize results
				// pathSmells := "results\\" + repoName + "\\" + prevCommit + "\\smells\\"
				// data := repoName + "," + prevCommit + "," + "Previous"

				// //design smells
				// sumDSmells := readSmells(pathSmells+"DesignSmells.csv", 3)
				// for _, smell := range designSmells {
				// 	data += "," + strconv.Itoa(sumDSmells[smell])
				// }

				// //implementation smells
				// sumISmells := readSmells(pathSmells+"ImplementationSmells.csv", 4)
				// for _, smell := range implSmells {
				// 	data += "," + strconv.Itoa(sumISmells[smell])
				// }
				data := summarizeSmells(repoName, prevCommit, "Previous", designSmells, implSmells)

				// respose time
				indCurr := -1
				for t := range times {
					if times[t].Commit == currCommit {
						indCurr = t
					}
				}
				if indCurr < 0 {
					fmt.Println("###### ERROR: Resptime of commit not found : ", currCommit)
				} else {
					oldTime := fmt.Sprintf("%f", times[indCurr].OldTime)
					data += "," + oldTime
					_, _ = datawriter.WriteString(data + "\n")
					datawriter.Flush()

					// //curr commit
					// pathSmells = "results\\" + repoName + "\\" + currCommit + "\\smells\\"
					// data = repoName + "," + currCommit + "," + "Current"

					// // design smells
					// sumDSmells = readSmells(pathSmells+"DesignSmells.csv", 3)
					// for _, smell := range designSmells {
					// 	data += "," + strconv.Itoa(sumDSmells[smell])
					// }
					// fmt.Println("design smells")
					// // implementation smells
					// sumISmells = readSmells(pathSmells+"ImplementationSmells.csv", 4)
					// for _, smell := range implSmells {
					// 	data += "," + strconv.Itoa(sumISmells[smell])
					// }
					// // fmt.Println("impl smells")
					data := summarizeSmells(repoName, currCommit, "Current", designSmells, implSmells)

					//time
					newTime := fmt.Sprintf("%f", times[indCurr].NewTime)
					data += "," + newTime
					_, _ = datawriter.WriteString(data + "\n")
					datawriter.Flush()
				}
			}
		} else {
			fmt.Println("Cannot clone repository: ", repoName)
		}

	}
	file.Close()
}

func processCommit(repoName, commit string) {
	fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repoName, repoName, commit)
	_, err := exec.Command("git", "--git-dir=repos\\"+repoName+"\\.git", "--work-tree=repos\\"+repoName, "checkout", commit).Output()
	if err != nil {
		fmt.Println("\nCannot run git checkout: ", err)
	}
	ProcessMetrics(repoName, commit)
	ProcessSmells(repoName, commit)
}

func summarizeSmells(repoName, commit, order string, designSmells, implSmells []string) string {
	//summarize results
	pathSmells := "results\\" + repoName + "\\" + commit + "\\smells\\"
	data := repoName + "," + commit + "," + order

	//design smells
	sumDSmells := readSmells(pathSmells+"DesignSmells.csv", 3)
	for _, smell := range designSmells {
		data += "," + strconv.Itoa(sumDSmells[smell])
	}

	//implementation smells
	sumISmells := readSmells(pathSmells+"ImplementationSmells.csv", 4)
	for _, smell := range implSmells {
		data += "," + strconv.Itoa(sumISmells[smell])
	}
	return data
}

func GetPreviousCommits(url, directory string, commits []string) map[string]string {
	var prevCommits = make(map[string]string)
	os.RemoveAll("temp\\")
	r, err := git.PlainClone("temp\\"+directory, false, &git.CloneOptions{URL: url})
	if err != nil {
		// log.Fatal(err)
		fmt.Println("Cannot clone repository: ", err)
	}
	if r != nil {
		ref, err := r.Head()
		if err != nil {
			//log.Fatal(err)
			fmt.Println("[ERROR]>> Cannot get Head commit of repository ", err)
		}
		if ref != nil {
			var prevCommit *object.Commit
			prevCommit = nil
			var prevTree *object.Tree
			prevTree = nil

			//fmt.Println(ref.Hash())
			cIter, err := r.Log(&git.LogOptions{From: ref.Hash()})
			if err != nil {
				//log.Fatal(err)
				fmt.Println("Cannot get log history of repository ", err)
			}
			err = cIter.ForEach(func(c *object.Commit) error {
				if prevCommit != nil {
					if prevTree != nil {
						hash := fmt.Sprintf("%v", c.Hash)
						prevHash := fmt.Sprintf("%v", prevCommit.Hash)
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
				fmt.Println(err)
			}
		}
	} else {
		fmt.Println("[ERROR]>> repository is nil.")
	}
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

func CloneRepo(url, folder string) bool {
	directory := "repos\\" + folder
	fmt.Printf("git clone -n %v repos\\%v\n", url, folder)

	r, err := git.PlainClone(directory, false, &git.CloneOptions{
		URL: url,
	})
	if err != nil {
		fmt.Println("Error cloning repository: ", err)
		r, err = git.PlainOpen(directory)
		if err != nil {
			fmt.Println("Error opening repository: ", err)
		}
	}
	if r != nil {
		return true
	} else {
		return false
	}
}

func ProcessMetrics(repo string, commit string) {
	path := "results\\" + repo + "\\" + commit + "\\metrics"
	checkDirectory(path)
	createUnderstandDb(repo, path)
	runUnderstand(path)
}

func createUnderstandDb(repo, path string) {
	_, err := exec.Command("und", "create", "-db", path+"\\understand.udb", "-languages", "java", "add", "repos\\"+repo).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Cannot create understand database: ", err)
	}
}

func runUnderstand(path string) {
	os.Remove("und.txt")
	cmd := path + `\\understand.udb
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
	path := "results\\" + repo + "\\" + commit + "\\smells"
	checkDirectory(path)
	runDesignite(repo, path)
}

func runDesignite(repo, path string) {
	_, err := exec.Command("java", "-jar", "DesigniteJava.jar", "-i", "repos\\"+repo, "-o", path).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Error trying to generate smells files: ", err)
	}
}

func checkDirectory(path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		dirs := strings.Split(path, "\\")
		subpath := ""
		for _, dir := range dirs {
			os.Mkdir(subpath+dir, 0755)
			subpath += dir + "\\"
		}
	}
}

func readSmells(path string, column int) map[string]int {
	fmt.Println(path)
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
		fmt.Println("Cannot read csv data 1: ", err)
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
	Commit  string
	OldTime float64
	NewTime float64
}

func readTime(path string) []StrTime {
	var mTime []StrTime
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
					ot, err := strconv.ParseFloat(row[1], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert OldTime to float", err)
					}
					nt, err := strconv.ParseFloat(row[2], 32)
					if err != nil {
						fmt.Println("### ERROR: Cannot convert NewTime to float", err)
					}

					t := StrTime{Commit: row[0], OldTime: ot, NewTime: nt}
					mTime = append(mTime, t)
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
