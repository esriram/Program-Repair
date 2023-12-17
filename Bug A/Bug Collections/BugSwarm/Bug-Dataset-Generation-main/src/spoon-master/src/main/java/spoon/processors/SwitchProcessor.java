// file processors/SwitchProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtSwitch;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class SwitchProcessor extends AbstractProcessor<CtSwitch> {
	public void process(CtSwitch element) {
		element.addComment(element.getFactory().Code().createComment("START OF SWITCH STATEMENT",CtComment.CommentType.INLINE));
	}
}
