import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_task(file, infoss):
    # gene geneLength mapped unmapped
    name = os.path.basename(file).replace(".stat", "")
    # print(name)
    results = {}
    count_dict = {}
    info_dict = {}
    f = open(file)
    lines = f.readlines()
    for line in lines:
        data = line.split('\t')
        all_data = data[0].split('~')
        acc_name = all_data[0]
        if data[0] != "*":
            gene_length = float(data[1])
            reads_count = float(data[2])
            total_reads = reads_dict[name]
            gene_name = data[0]
            if gene_length > 0 and reads_count > 0:
                rpkm = reads_count * 1000000 / float(total_reads) * 1000 / gene_length
                if gene_name not in info_dict:
                    info_dict[gene_name] = name + '\t' + gene_name
                if gene_name not in count_dict:
                    count_dict[gene_name] = 0
                count_dict[gene_name] += rpkm
    f.close()
    for gene_name, rpkm in count_dict.items():
        if name not in results:
            results[name] = []
        results[name].append(info_dict[gene_name] + '\t' + str(rpkm))
    return results

def parallel_task(task_list, infoss, threads_num=20, interval=0):
    executor = ThreadPoolExecutor(max_workers=threads_num)
    result = []
    futures = []
    for i, task in enumerate(task_list):
        future = executor.submit(execute_task, task, infoss)
        time.sleep(interval)
        futures.append(future)
    for future in as_completed(futures):
        status = future.done()
        data = future.result()
        result.append(data)
    executor.shutdown(True)
    return result

def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Main program
if __name__ == "__main__":
    fqfile = os.path.abspath(sys.argv[1])  # reads count file
    resultdir = os.path.abspath(sys.argv[2])
    outfile = os.path.abspath(sys.argv[3])

    makedir(resultdir)
    outdir = os.path.join(resultdir)

    reads_dict = {}
    with open(fqfile) as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            name = os.path.basename(data[0]).replace("_1.fastq.gz", "")
            reads_dict[name] = float(data[-1])  # Read reads_count from the last column

    task_list = [os.path.join(outdir, file) for file in os.listdir(outdir) if re.search("stat$", file)]
    all_results = parallel_task(task_list, reads_dict, threads_num=1)

    with open(outfile, 'w') as w:
        w.write("Sample\tGenename\tRPKM\n")
        for single_result in all_results:
            for name, result_list in single_result.items():
                for results1 in result_list:
                    w.write(results1 + '\n')