// file processors/WhileProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtWhile;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class WhileProcessor extends AbstractProcessor<CtWhile> {
	public void process(CtWhile element) {
		element.addComment(element.getFactory().Code().createComment("START OF WHILE LOOP STATEMENT",CtComment.CommentType.INLINE));
	}
}
