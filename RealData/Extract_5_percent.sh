 # Code to only use the SNPs with more than 5% frequency

bcftools +fill-tags all_chr.biallelic.every2k.vcf.gz -Ou -- -t AF | \
bcftools view -i 'AF>=0.05 && AF<=0.95' -Oz -o all_chr.biallelic.every2k.maf5.vcf.gz

bcftools index -f all_chr.biallelic.every2k.maf5.vcf.gz
