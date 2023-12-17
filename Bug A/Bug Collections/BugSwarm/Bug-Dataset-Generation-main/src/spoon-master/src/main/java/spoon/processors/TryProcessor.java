// file processors/TryProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtTry;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class TryProcessor extends AbstractProcessor<CtTry> {
	public void process(CtTry element) {
		element.addComment(element.getFactory().Code().createComment("START OF TRY STATEMENT",CtComment.CommentType.INLINE));
	}
}
