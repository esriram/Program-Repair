// file processors/IfProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtIf;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class IfProcessor extends AbstractProcessor<CtIf> {
	public void process(CtIf element) {
		element.addComment(element.getFactory().Code().createComment("START OF IF STATEMENT",CtComment.CommentType.INLINE));
	}
}
