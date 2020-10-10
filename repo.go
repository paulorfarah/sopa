package main

import (
	"log"
	"fmt"
	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
)

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

func main() {
	url := "http://github.com/paulorfarah/refactoring-python-code"
	dir := "repos"
	commits := []string{"3599f5cdc72e5526de48df98c212322a835869cd", "ee950eac81d01d30b473d0a0aa74d7a94e22a8e6", "a74d09e3824c152fa0348a9f36e5b2c8af27a181"}
	res := GetPreviousCommits(url, dir, commits)
	for k, v := range res {
		fmt.Printf("%v: %v\n", k, v)
	}
}
