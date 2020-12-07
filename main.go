package main

//	"fmt"

func main() {
	// urlsPeass := map[string]string{
	// 	"commons-compress": "https://github.com/apache/commons-compress",
	// 	"commons-csv":        "https://github.com/apache/commons-csv",
	// 	"commons-dbcp":       "https://github.com/apache/commons-dbcp",
	// 	"commons-fileupload": "https://github.com/apache/commons-fileupload",
	// 	"commons-imaging":    "https://github.com/apache/commons-imaging",
	// 	"commons-io":         "https://github.com/apache/commons-io",
	// 	"commons-pool":       "https://github.com/apache/commons-pool",
	// 	"commons-text":       "https://github.com/apache/commons-text",
	// 	"jackson-core":       "https://github.com/FasterXML/jackson-core",
	// 	"k-9":                "https://github.com/k9mail/k-9",
	// }

	// urlHadoop := map[string]string{
	// 	"hadoop": "https://github.com/apache/hadoop",
	// }

	urls := map[string]string{
		// "okhttp": "https://github.com/square/okhttp",
		"hadoop": "https://github.com/apache/hadoop",
	}

	ParseHadoopResults(urls)
	// ParsePeassResults()
	// ParseTravisTorrent()
	SummarizeResults()
	ReadSmellsFromCommits(urls)

	for k := range urls {
		path := "results/sum/sum_" + k + ".csv"
		ResptimeTravis(k, path)
	}

}
