#!/bin/bash
timestamp() {
  date +"%s"
}
fname="$(timestamp).csv"

# send csv to server
echo "Compiling..."
./compile

echo "Running solver..."
./test >> $fname

echo "Sending data to data representation server"
scp "./$fname" "root@github.space:~/clojurescript-ode-solvers/static-site/data/$fname"
