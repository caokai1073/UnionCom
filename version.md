## v0.1.6
+ Software optimization.
+ Change Euclidean distance to geodesic distance in KL divergence.

## v0.2.0
+ Software optimization.
+ Split function train into functions Match and Project.
+ Use Kuhn-Munkres to find optimal pairs between datasets instead of parbabilistic matrix matching.
+ Add a new parameter "project" to provide options for barycentric projection.

### v0.2.1
+ Software optimization
+ Split function "train" into functions "Match" and "Project"
+ Use Kuhn-Munkres algorithm to find optimal pairs between datasets instead of parbabilistic matrix matching
+ Add a new parameter "project" to provide options for barycentric projection
+ Separate "test_label_transfer_accuracy" function from "fit_transform" function
+ fix some bugs

## v0.2.2
+ Fix some bugs;
+ Change function "PCA_visualize" to "Visualize", and provide PCA, TSNE and UMAP for visulalization;
+ Add a function to find maximum connected subgraph;

## v0.3.0
+ Add more comments and make the software easier to understand;
+ Fix some bugs;