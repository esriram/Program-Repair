// file processors/ControlFlowProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.CtModelImpl.CtRootPackage;
import spoon.reflect.factory.Factory;
import fr.inria.controlflow.ControlFlowBuilder;
import fr.inria.controlflow.ControlFlowGraph;

import java.io.*;

public class ControlFlowProcessor extends AbstractProcessor<CtElement> {
	public void process(CtElement element) {
		System.out.println("Hi");
		ControlFlowBuilder builder = new ControlFlowBuilder();
    		//ControlFlowGraph graph = builder.build(element);
    		//System.out.println(graph.toGraphVisText());
	}
}
