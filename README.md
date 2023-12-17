# Program-Repair

Project Structure

Bug A/Algo/Repair Models - Change datasets for each repair model to test performance.

Bug A/Bug Collections/Datasets - Standard Bug Datasets that models are tested upon.


With automatic validation, the ability of program repair models in finding and fixing bugs in new projects is zero. 

For example, consider this example bug with BugSwarm Dataset, 
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\CheckBoxButtonGwt.java)
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\InputToggleButtonGwt.java)
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\RadioButtonGwt.java)

The problem with the above code in CheckBoxButtonGwt.java is that, the import statement of 

import org.semanticweb.owlapi.reasoner.InconsistentOntologyException; is missing. These models are not able to fix such bugs that require knowledge of cross-file reference in a project.

Another example from RegMiner, (codehaus_jettison/1/ric/src/main/java/org/codehaus/jettison/mapped/Configuration.java), the recommended fix is the addition of data variable 'ignoreEmptyArrayValues' and a getter and setter methods.

> private boolean ignoreEmptyArrayValues;
> 
> public boolean isIgnoreEmptyArrayValues() {
> 
>        return ignoreEmptyArrayValues;
>
> }
>
> public void setIgnoreEmptyArrayValues(boolean ignoreEmptyArrayValues) {
>
>       this.ignoreEmptyArrayValues = ignoreEmptyArrayValues;
> }

However, this requires knowledge from files that use this class to understand that a data variable is missing. These models are not equipped for these.

These models perform better when the fixes are local to a file and have logical binding to function in test. Examples are buggy files in QuixBugs and Defects4J. 

>def bitcount(n):
>  count = 0
>  while n:
>    n ^= n - 1
>    count += 1
>  return count

The fixed version for counting 1-bits is to replace ^ operator with &, that is, n &= n - 1 inside the while loop. These are the kind of bugs the models can perform and repair.

However, when we have complex project-structured bugs, the models do not perform. 

# Why is manual validation giving better numbers?

With manual validation of program repair models against standard bug datasets,

Projects | SelfAPR	   | FitRepair	    |AlphaRepair	|  ITER	 |       MUFIN
---|---|---|---|---|---
D4J | 0.722222222|	0.788888889	|  0.8	     |     0.766666667	|  0.8
Quix Bugs (Java)	|  0.766666667|	0.833333333	|  0.866666667	 | 0.683333333	|  0.85
Quix Bugs (Python)|	0.833333333|	0.766666667	 | 0.9	    |      0.65	   |     0.866666667
Many Bugs	    |      0.5	   |     0.789473684|	  0.684210526	|  0.578947368	 | 0.631578947
BugSwarm	     |     0.632653061|	0.87755102	|  0.836734694	|  0.795918367	|  0.734693878
RegMiner	 |         0.675	  |    0.8	      |    0.7	      |    0.725	  |      0.775

Better numbers for manual validation is the result of comparison between 'fixed' and 'expected' version of all files in a project. The projects are generally having a number of files that do not have any modification required. For example, in a project named 'codehaus_jettison' in RegMiner, the number of files requiring modification is 11, while the total number of files is more than 50. Therefore, the other 39 files would pass, resulting in 39/50 rate of successful files. But, upon setting automatic running of test, the project would not pass. 

