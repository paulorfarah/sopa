package main

import (
	"io"
	"os"
	"log"
	"fmt"
	"strings"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"encoding/csv"
	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
//	"github.com/tobgu/qframe"
)

func ReadCommits() {
	urls := map[string]string {
		"commons-compress":"https://github.com/apache/commons-compress",
		"commons-csv":"https://github.com/apache/commons-csv",
		"commons-dbcp":"https://github.com/apache/commons-dbcp",
		"commons-fileupload":"https://github.com/apache/commons-fileupload",
		"commons-imaging":"https://github.com/apache/commons-imaging",
		"commons-io":"https://github.com/apache/commons-io",
		"commons-pool":"https://github.com/apache/commons-pool",
		"commons-text":"https://github.com/apache/commons-text",
		"jackson-core":"https://github.com/FasterXML/jackson-core",
		"k-9":"https://github.com/k9mail/k-9",
	}

	dir := filepath.FromSlash("results/sum/")
	//open file
	abs, err := filepath.Abs(dir)
	if err != nil {
		log.Fatal("Cannot read absolute filepath", err)
	}
	files := getFiles(abs, ".csv")
	for _, filename := range files {
		repoName := strings.ReplaceAll(filename, ".csv", "")
		repoName = strings.ReplaceAll(repoName, "sum_peass_", "")
		CloneRepo(urls[repoName], repoName)

		sumFile, err := os.Open(dir + filename)
		if err != nil {
			log.Fatal(err)
		}
		//f := qframe.ReadCSV(sumFile)
		//fmt.Println("dataframe...")
		//fmt.Println(f)

		//viewCommits := f.MustStringView("commit")
		//for i := 0; i < viewCommits.Len(); i++ {
		//	com := fmt.Sprintf("%x", viewCommits.ItemAt(i))
		//	ProcessMetrics(repoName, com)
		//}

		r := csv.NewReader(sumFile)
		var commits []string
		for {
			commit, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("Commit: %s\n", commit[0])
			if commit[0] != "commit" {
				commits = append(commits, commit[0])
			}
		}
		prevCommits := GetPreviousCommits(urls[repoName], repoName, commits)
		for currCommit, prevCommit := range prevCommits {
			fmt.Printf("curr: %s, prev: %s\n", currCommit, prevCommit)
			//curr commit
			fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repoName, repoName, currCommit)
			_, err := exec.Command("git", "--git-dir=repos\\"+repoName + "\\.git", "--work-tree=repos\\"+repoName, "checkout", currCommit).Output()
			if err != nil {
				fmt.Println("\nCannot run git checkout: ", err)
			}
			ProcessMetrics(repoName, currCommit)
			ProcessSmells(repoName, currCommit)

			// previous commit
			fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repoName, repoName, prevCommit)
			_, err = exec.Command("git", "--git-dir=repos\\"+repoName + "\\.git", "--work-tree=repos\\"+repoName, "checkout", prevCommit).Output()
			if err != nil {
				fmt.Println("\nCannot run git checkout: ", err)
			}
			ProcessMetrics(repoName, prevCommit)
			ProcessSmells(repoName, prevCommit)

		}
	}
}

func GetPreviousCommits(url string, directory string, commits []string) map[string]string {
	var prevCommits = make(map[string]string)

	r, err := git.PlainClone(directory, false, &git.CloneOptions{
		URL: url,
	})

	if err != nil {
		log.Fatal(err)
	}
	if r != nil {
		ref, err := r.Head()
		if err != nil {
			log.Fatal(err)
		}
		if ref != nil {
			var prevCommit *object.Commit
			prevCommit = nil
			var prevTree *object.Tree
			prevTree = nil

			//fmt.Println(ref.Hash())
			cIter, err := r.Log(&git.LogOptions{From: ref.Hash()})
			if err != nil {
				log.Fatal(err)
			}
			err = cIter.ForEach(func(c *object.Commit) error {
				if prevCommit != nil {
					if prevTree != nil {
						hash := fmt.Sprintf("%v", c.Hash)
						prevHash := fmt.Sprintf("%v", prevCommit.Hash)
						if findCommit(commits, hash) == true {
							prevCommits[prevHash] = hash
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

func CloneRepo(repository, folder string) {
	fmt.Println("clone repository: ", repository)
	fmt.Printf("git clone -n %v repos\\%v\n", repository, folder)
	_, err := exec.Command("git", "clone", "-n", repository, "repos\\" + folder).Output()
	if err != nil {
		fmt.Println("Error clonning repo: ", err)
	}
}

func ProcessMetrics(repo string, commit string) {
	path := "results\\" + repo + "\\" + commit + "\\metrics"
	checkDirectory(path)
	createUnderstandDb(repo, path)
	runUnderstand(path)
}

func createUnderstandDb(repo, path string) {
	_, err := exec.Command("und", "create", "-db", path + "\\understand.udb", "-languages", "java", "add", "repos\\" + repo).Output()
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

	fmt.Printf("und process und.txt\n")
	_ , err = exec.Command("und", "process", "und.txt").Output()

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
	_, err := exec.Command("java", "-jar", "DesigniteJava.jar", "-i", "repos\\" + repo, "-o", path).Output()
	if err != nil {
		fmt.Println("[ERROR]>> Error trying to generate smells files: ", err)
	}
}

func checkDirectory(path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		dirs := strings.Split(path, "\\")
		subpath := ""
		for _, dir := range dirs {
			os.Mkdir(subpath + dir, 0755)
			subpath += dir + "\\"
		}
	}
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
