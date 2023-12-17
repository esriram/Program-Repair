// file processors/ForEachProcessor.java
package spoon.processors;

import spoon.processing.AbstractProcessor;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.reflect.visitor.CtIterator;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.code.CtForEach;
import spoon.reflect.code.CtComment;
import spoon.reflect.factory.Factory;

import java.util.List;
import java.io.*;

public class ForEachProcessor extends AbstractProcessor<CtForEach> {
	public void process(CtForEach element) {
		element.addComment(element.getFactory().Code().createComment("START OF FOR-EACH LOOP STATEMENT",CtComment.CommentType.INLINE));
	}
}
