for report in kraken2_out/*_report.tsv
do
    sample=$(basename "$report" _report.tsv)
    bracken -d /public/database/kraken2_PF20260226 \
            -i "$report" \
            -o bracken/genus/"$sample".bracken \
            -w bracken/genus/"$sample".bracken.report \
            -r 150 \
            -l G \
            -t 10
done