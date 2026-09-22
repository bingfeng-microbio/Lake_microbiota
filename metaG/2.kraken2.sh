#!/bin/bash

# db
db_kraken2="/public/database/kraken2_PF20260226"

input_path=$1
output_path=$2
num_jobs=$3

# Check if the correct number of arguments is provided
if [ $# -ne 3 ]; then
echo "Usage: $0 <input_path> <output_path> <num_jobs>"
exit 1
fi

# Function to process a single file
process_kraken2() {
local F="$1"

R=${F%_*}_2.fastq.gz
BASE=${F##*/}
SAMPLE=${BASE%_*}
echo $SAMPLE

if [ -e $output_path/${SAMPLE}_report.tsv ]; then
    echo "$SAMPLE SKIP"
else
    kraken2 --db "$db_kraken2" --threads 30 --report "$output_path/${SAMPLE}_report.tsv" --output "$output_path/${SAMPLE}_out.tsv" --use-names --report-zero-counts --paired $F $R
    echo "$SAMPLE kraken2 DONE"
fi

}

## Use parallel to process gzip files in parallel with specified number of jobs
for F in $input_path/*_1.fastq.gz; do
process_kraken2 "$F" &
((++processed_files))
[ $((processed_files % num_jobs)) -eq 0 ] && wait
done
