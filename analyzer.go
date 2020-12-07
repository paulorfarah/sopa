package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func ResptimeTravis(repo, path string) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = repo
	p.X.Label.Text = "Commits"
	p.Y.Label.Text = "Response time"

	xys := readData(path)
	err = plotutil.AddScatters(p,
		"Resptime", xys)

	if err != nil {
		log.Fatalf("Cannot read sum result file...")
	}

	// Save the plot to a PNG file.
	if err := p.Save(16*vg.Inch, 8*vg.Inch, repo+".png"); err != nil {
		panic(err)
	}
}

func readData(file string) plotter.XYs {
	var x []float64
	var y []float64

	csvfile, err := os.Open(file)
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)
	//r := csv.NewReader(bufio.NewReader(csvfile))

	// Iterate through the records

	for count := float64(1); ; count++ {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			vi
			log.Fatal(err)
		}
		fmt.Printf("%d) commit: %s resptime %s\n", count, record[0], record[3])
		// "commit", "method", "oldTime", "currTime", "diffTime", "changePercent"
		x = append(x, count) //record[0])
		f, err := strconv.ParseFloat(record[3], 64)
		y = append(y, f)

	}

	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}
