
About
=====
The ManySStuBs4J corpus is a collection of simple fixes to Java bugs, designed for evaluating program repair techniques.
We collect all bug-fixing changes using the SZZ heuristic, and then filter these to obtain a data set of small bug fix changes.
These are single statement fixes, classified where possible into one of 16 syntactic templates which we call SStuBs.
The dataset contains simple statement bugs mined from open-source Java projects hosted in GitHub.
There are two variants of the dataset. One mined from the 100 Java Maven Projects and one mined from the top 1000 Java Projects.
A project's popularity is determined by computing the sum of z-scores of its forks and watchers.
We kept only bug commits that contain only single statement changes and ignore stylistic differences such as spaces or empty as well as differences in comments.
Some single statement changes can be caused by refactorings, like changing a variable name rather than bug fixes.
We attempted to detect and exclude refactorings such as variable, function, and class renamings, function argument renamings or changing the number of arguments in a function.
The commits are classified as bug fixes or not by checking if the commit message contains any of a set of predetermined keywords such as bug, fix, fault etc.
We evaluated the accuracy of this method on a random sample of 100 commits that contained SStuBs from the smaller version of the dataset and found it to achieve a satisfactory 94% accuracy.
This method has also been used before to extract bug datasets (Ray et al., 2015; Tufano et al., 2018) where it achieved an accuracy of 96% and 97.6% respectively.

The bugs are stored in a JSON file (each version of the dataset has each own instance of this file).
Any bugs that fit one of 16 patterns are also annotated by which pattern(s) they fit in a separate JSON file (each version of the dataset has each own instance of this file).
We refer to bugs that fit any of the 16 patterns as simple stupid bugs (SStuBs).

For more information on extracting the dataset and a detailed documentation of the software visit our GitHub repo: https://github.com/mast-group/SStuBs-mining


Files
=====
100 Java Maven Project Bugs				bugs (bugs.json)
1000 Java Project Bugs					bugsLarge (bugsLarge.json)
100 Java Maven Project SStuBs				sstubs (sstubs.json)
1000 Java Project SStuBs				sstubsLarge (sstubsLarge.json)
Top 100 Ranked Java Maven Projects			topJavaMavenProjects.csv
Ranked Java Projects					topProjects.csv

Due to a bug zenodo returns an error when uploading json files.
The .json suffix can be restored by simply renaming the files (e.g. bugs -> bugs.json).



Corpus Statistics
=================
-------------------------------------------------------------------------------------------------------------------------
|   Projects            Bug Commits         Buggy Statements       Bug Statements per Commit   	   Pattern Fitting Bugs |
+-----------------------------------------------------------------------------------------------------------------------+
| 100 Java Maven  	   12598     	        25539      	           2.03      		           7824         |
| 1000 Java      	   86771     	       153652      	           1.77       		           51537        |
+-----------------------------------------------------------------------------------------------------------------------+


The table below provides information about the number of mined single statement bugs that fit each of the SStuB patterns.
-------------------------------------------------------------------------
|	Pattern Name		Instances	Instances Large     	|
+-----------------------------------------------------------------------+
| Change Identifier Used  	   3265		      22668      	|
| Change Numeric Literal	   1137   	      5447       	|
| Change Boolean Literal	   169	  	      1842       	|
| Change Modifier       	   1852   	      5011       	|
| Wrong Function Name   	   1486   	      10179      	|
| Same Function More Args	   758   	      5100       	|
| Same Function Less Args	   179   	      1588       	|
| Same Function Change Caller	   187   	      1504       	|
| Same Function Swap Args	   127   	      612       	|
| Change Binary Operator	   275   	      2241       	|
| Change Unary Operator		   170   	      1016       	|
| Change Operand        	   120   	      807       	|
| Less Specific If      	   215   	      2813       	|
| More Specific If      	   175   	      2381       	|
| Missing Throws Exception	   68   	      206       	|
| Delete Throws Exception	   48   	      508       	|
+-----------------------------------------------------------------------+


Use
=====
The ManySStuBs4J Corpus is an automatically mined collection of Java bugs at large scale.
We note that the automatic extraction could have inserted some noise. 
However, the amount of inserted noise is deemed to be almost negligible (see about).
We also note that the code of the Java projects is not ours but is open-source. 
Please respect the license of each project.

If you use the data set in a research publication, please cite:

@inproceedings{ManySStuBs4JCorpus2020,
	author={Karampatsis, Rafael-Michael and Sutton, Charles},
	title={{How Often Do Single-Statement Bugs Occur?\\ The ManySStuBs4J Dataset}},
	booktitle={},	
	year={2020},
	pages={},
	organization={}
}

JSON Fields
===========
The files sstubs.json and sstubsLarge.json contain the following fields:

"bugType"		:	The bug type (16 possible values).
"commitSHA1"		:	The hash of the commit fixing the bug.  
"fixCommitParentSHA1"	:	The hash of the last commit containing the bug.
"commitFile"		:	Path of the fixed file.
"patch"  		:	The diff of the buggy and fixed file containing all the changes applied by the fix commit.
"projectName"		:	The concatenated repo owner and repo name separated by a '.'.
"bugLineNum"		:	The line in which the bug exists in the buggy version of the file.
"bugNodeStartChar"	:	The character index (i.e., the number of characters in the java file that must be read before encountering the first one of the AST node) at which the affected ASTNode starts in the buggy version of the file. 
"bugNodeLength"		:	The length of the affected ASTNode in the buggy version of the file.
"fixLineNum"		:	The line in which the bug was fixed in the fixed version of the file.
"fixNodeStartChar"	:	The character index (i.e., the number of characters in the java file that must be read before encountering the first one of the AST node) at which the affected ASTNode starts in the fixed version of the file.
"fixNodeLength"		:	The length of the affected ASTNode in the fixed version of the file.
"before"		:	The affected AST's tree (sometimes subtree  e.g. Change Numeric Literal) text before the fix.
"after"			:	The affected AST's tree (sometimes subtree  e.g. Change Numeric Literal) text after the fix. 

The "before", "after", "patch" fields help humans to understand the change.
The "bugLineNum", "bugNodeStartChar", "bugNodeLength", "fixLineNum", "fixNodeStartChar", and "fixNodeLength" allow pinpointing of the AST nodes and lines that contained the bug and their equivalent ones in the  fixed version of the file.

Similarly the bugs in bugs.json contain the above fields except bugType.
All bugs appearing in sstubs.json have also an entry in bugs.json.

