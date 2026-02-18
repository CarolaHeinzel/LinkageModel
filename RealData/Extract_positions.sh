#Code to extract the positions from the SNPS

#!/bin/bash

input="all_chr.biallelic.every2k.maf5.vcf.gz"
output="snp_output.txt"

# Extract CHROM and POS and generate artificial rsID = chr<CHR>_<POS>
bcftools query -f '%CHROM\t%POS\n' "$input" | \
awk '{
    printf("{'\''rsID'\'': '\''chr%s_%s'\'', '\''Chr'\'': %s, '\''Position'\'': %s}\n",$1,$2,$1,$2)
}' > "$output"

echo "Done. Output written to $output"
