 

#!/bin/bash

for i in {1..22}
do
    input="ALL.chr${i}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    output="chr${i}.biallelic.every2k.vcf.gz"

    # 1. Extract biallelic SNPs (uncompressed BCF stream)
    bcftools view \
        --types snps \
        --min-alleles 2 \
        --max-alleles 2 \
        -Ou "$input" | \

    # 2. Convert to VCF and keep header + every 2000th variant line
    bcftools view -Ov | \
    awk 'BEGIN{c=0}
         /^#/ {print; next}
         {c++; if(c%2000==0) print}' | \
    bgzip -c > "$output"

    # 3. Index output
    bcftools index -f "$output"
done


 

bcftools concat \
    -Oz \
    -o all_chr.biallelic.every2k.vcf.gz \
    chr*.biallelic.every2k.vcf.gz
