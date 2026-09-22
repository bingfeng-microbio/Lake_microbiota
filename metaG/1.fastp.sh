#!/bin/bash

inputpath=$1
outputpath=$2
max_parallel=$3

# Function to process a single gzip file
process_fastp() {
    local F="$1"
    r=${F##*/}
    name=${r%_*}
    echo "Trim: ${name}"
    fastp -i ${F} \
    -I ${inputpath}/${name}_2.fastq.gz \
    -o ${outputpath}/${name}_1.fastq.gz \
    -O ${outputpath}/${name}_2.fastq.gz \
    -h ${outputpath}/${name}.html \
    -j ${outputpath}/${name}.json \
    --thread 16
}

# Use '&' to run the process_fastp function in background
count=0
for F in ${inputpath}/*_1.fastq.gz; do
    process_fastp "$F" &
    ((count++))
    if ((count >= max_parallel)); then
        wait
        count=0
    fi
done

# Wait for the remaining background jobs to finish
wait

