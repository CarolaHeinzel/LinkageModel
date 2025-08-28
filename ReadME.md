# LinkageModel

This repository contains the following python files. <br>

Grid_Search_Linkage.py: Code to calculate the MLE for the ancestry in the Linkage Model for arbitrary many chromosomes. <br>
Test_whole_Population.py : Code to apply the statistical test with H0: The user should apply the Admixture Model vs. H1: The user should use the Linkage Model to real data.<br>
Fischer_Information_diploid.py: Fischer Information for the Admixture Model. <br>

Additionally, it contains <br>

1000G_AIMsetKidd.vcf: example data, which have the correct format for the statistical test. These are the 1000 Geneomes Data taken from [the GitHub page from Peter Pfaffelhuber](https://github.com/pfaffelh/recent-admixture/blob/master/data/1000G/1000G_AIMsetKidd.vcf.gz).<br>

The folder Evaluation consists of code that has been used to evaluate the methods. This includes
: Code to evaluate the statistical test.<br>
: Code to plot the results.<br>
: Code to evaluate the code to find the MLE in the Linkage Model.<br>
Bootstrap_TrueData.py: Code to compare the  Fischer Information to the bootstrap variance. <br>
: Code to calculate the Fischer Information for both, the linkage model and the admixture model.<br>
cM_calculateion.py: Code to calculate the differences between two loci in centi-Morgan. <br>

## Application of the code

To apply the code, please download the files and run them in e.g. spyder. <br>

## Example Output of the code

The example output of the code can be viewed as a comment in the code.<br>
