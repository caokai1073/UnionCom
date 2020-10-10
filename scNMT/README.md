+ The full data is in the links https://drive.google.com/drive/folders/1SMexG_zTZUr5Vt9Ua9_NMngunfwUpdK6?usp=sharing

We obtained the scNMT-seq data "gastrulation_scnmt_mofa.RData" from ftp://ftp.ebi.ac.uk/pub/databases/scnmt_gastrulation. We are grateful to Dr. Stephen Clark for providing the parsed data. But it seems that they have updated their data in this link. So I attach the origin  "gastrulation_scnmt_mofa.RData"  data in google drive. ("gastrulation_scnmt_mofa.RData"). 

Data pre-proecessing: We use the Promoter accessibility (1940 cells with 2500 features), Promoter methylation (1940 cells with 2500 features) and RNA expression (1940 cells with 5000 features) in MOFAmodel/data/ of gastrulation_scnmt_mofa.RData. (See "Promoter_accessibility.txt, Promoter_methylation.txt, RNA.txt")

However, the data are too many missing values  ('NA')  in accessibility and methylation.  We did the following three steps 
1) We replaced the 'NA' values with 0.   
2) We then filtered out cells with all features being denoted as 0, resulting in 1940 cells with 5000 features in RNA expression, 709 cells with 2500 features in the methylation and 612 cells with 2500 features in the accessibility. 
3) Afterwards, we find the UMAP can obtain a best perservation of the global structure of cell lineage for this dataset than PCA and t-SNE. Therefore, we applied UMAP to conduct the dimensionality reduction of each of the 3 dataset to a dimensionality of 300, respectively, prior to the alignment. 
(See attached  "proecess.py",  "Paccessibility_300.txt, Pmethylation_300.txt, RNA_300.txt")

After that, the input matrices become 612\times 300 for accessibility, 709\times 300 for methylation and 1940\times 300 for RNA expression. The alignment of accessibility and methylation is relatively easy because they share most of the cells (See scNMT/process.pu and scNMT/match.txt). However, because RNA expression has 1940 cells, much more than the other two sets,  the alignmnet of 3 datasets indeed needs some tricks
1) We chose the maximum connected subgraph of RNA expression data. (line173-178 in "UnionCom.py")
2) We used a MinMaxScaler normalization for RNA expression data. (line 179-180 in attached "UnionCom.py")
The result of alignment are shown in "result/". 

In the original version, we did not use MOFA's built-in methods for preprocessing data, but we found it is a more efficient way. The "Eaccess_MOFA.txt" and "Emethy_MOFA.txt" in Github were preprocessed by MOFA built-in methods.  The numbers in "type1.txt" and "type2.txt" mean different cell stages. ''0" represents "E5.5", '1" represents "E6.5".'2" represents "E7.5" (See "proecess.py" for the generation of "type1.txt" and "type2.txt").

In the scNMT-seq data we obtained, the authors of scNMT-seq paper did not provide the "E4.5" data. But they have already updated new data. According to my observation, the RNA expression is the most difficult to match.
