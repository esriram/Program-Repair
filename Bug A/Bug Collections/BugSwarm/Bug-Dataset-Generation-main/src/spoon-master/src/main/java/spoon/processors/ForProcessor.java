// file processors/ForProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtFor;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class ForProcessor extends AbstractProcessor<CtFor> {
	public void process(CtFor element) {
		element.addComment(element.getFactory().Code().createComment("START OF FOR LOOP STATEMENT",CtComment.CommentType.INLINE));
	}
}
