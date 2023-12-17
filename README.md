# Program-Repair

With manual validation of program repair models against standard bug datasets,

Projects	          SelfAPR	    FitRepair	    AlphaRepair	  ITER	        MUFIN

D4J	                0.722222222	0.788888889	  0.8	          0.766666667	  0.8

Quix Bugs (Java)	  0.766666667	0.833333333	  0.866666667	  0.683333333	  0.85

Quix Bugs (Python)	0.833333333	0.766666667	  0.9	          0.65	        0.866666667

Many Bugs	          0.5	        0.789473684	  0.684210526	  0.578947368	  0.631578947

BugSwarm	          0.632653061	0.87755102	  0.836734694	  0.795918367	  0.734693878

RegMiner	          0.675	      0.8	          0.7	          0.725	        0.775

With automatic validation, the ability of these models in finding and repairing bugs is zero. 

For example, consider this example bug with BugSwarm Dataset, 
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\CheckBoxButtonGwt.java)
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\InputToggleButtonGwt.java)
(bugswarm\data\gwtbootstrap3\2\fail\gwtbootstrap3\src\main\java\org\gwtbootstrap3\client\ui\RadioButtonGwt.java)

The problem with the above code in CheckBoxButtonGwt.java is that, 
The import statement of import org.semanticweb.owlapi.reasoner.InconsistentOntologyException; is missing. These models are not able to fix such bugs that require knowledge of cross-file reference in a project.

Another example, 
