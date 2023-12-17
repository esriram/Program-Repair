// file processors/CatchProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtCatch;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class CatchProcessor extends AbstractProcessor<CtCatch> {
	public void process(CtCatch element) {
		element.addComment(element.getFactory().Code().createComment("START OF CATCH STATEMENT",CtComment.CommentType.INLINE));
	}
}
