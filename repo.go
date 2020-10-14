package main

import (
	"io"
	"os"
	"log"
	"fmt"
	"strings"
	"os/exec"
	"path/filepath"
	"encoding/csv"
	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
//	"github.com/tobgu/qframe"
)

func ReadCommits() {
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
		CloneRepo(repoName)
		CreateUnderstandDb(repoName)
		
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
		for {
			commit, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("Commit: %s\n", commit[0])
			ProcessMetrics(repoName, commit[0])
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

func CloneRepo(repository string) {
	fmt.Println("clone repository: ", repository)
	fmt.Printf("git clone -n https://github.com/apache/%v repos\\%v\n", repository, repository)
	_, err := exec.Command("git", "clone", "-n", "https://github.com/apache/" + repository, "repos\\" + repository).Output()
	if err != nil {
		//log.Fatal(err)
		fmt.Println("Error clonning repo: ", err)
	}
	//fmt.Printf("Clonning result: %s\n", out)
}

func CreateUnderstandDb(repo string) {
	_, err := exec.Command("und", "create", "-db", "results\\und\\" + repo + ".udb", "-languages", "java").Output()
	if err != nil {
		fmt.Println("Error processing metrics: ", err)
	}
//	fmt.Printf("Clonning result: %s\n", out)

}

func ProcessMetrics(repo string, commit string) {
	fmt.Printf("git --git-dir=repos\\%v\\.git --work-tree=repos\\%v checkout %s\n", repo, repo, commit)
	//s := fmt.Sprintf("%x", commit)
	_, err := exec.Command("git", "--git-dir=repos\\"+repo + "\\.git", "--work-tree=repos\\"+repo, "checkout", commit).Output()
	if err != nil {
		fmt.Println("\nError processing metrics: ", err)
	}

	RunUnderstand(repo)
}

func RunUnderstand(repo string) {
	//add repository
	fmt.Printf("und -db results\\und\\%v.udb add repos\\%v\n",repo, repo)
	_, err := exec.Command("und", "-db", "results\\und\\" + repo + ".udb", "add", "repos\\" + repo).Output()
	if err != nil {
		fmt.Println("[>>Error]: Cannot add repository path: ", err)
	}

	//add metrics settings
	fmt.Printf("und -MetricMetrics \"AvgCyclomatic\" settings results\\und\\%v.udb\n",repo)
	_, err = exec.Command("und", "-MetricMetrics", "\"AvgCyclomatic\"", "settings", "results\\und\\" + repo + ".udb").Output()
	if err != nil {
		fmt.Println("[>>Error]: Cannot add metric to understand db: ", err)
	}

	//metrics
	fmt.Printf("und metrics results\\und\\%v.udb\n",repo)
	_, err = exec.Command("und", "metrics", "results\\und\\" + repo + ".udb").Output()
	if err != nil {
		fmt.Println("[>>Error]: Error trying to generate metrics file: ", err)
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
