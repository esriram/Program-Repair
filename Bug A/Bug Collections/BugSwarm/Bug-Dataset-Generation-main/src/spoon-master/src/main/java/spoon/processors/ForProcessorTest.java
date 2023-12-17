// file processors/ForProcessorTest.java
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

/**
 * Add comment when for loop (not forEach) starts and ends
 */
public class ForProcessorTest extends AbstractProcessor<CtElement> {
	public boolean lastElementIsFor = false;
	public TypeFilter filter = new TypeFilter<CtFor>(CtFor.class);
	public void process(CtElement element) {
		if(filter.matches(element)){
			lastElementIsFor = true;
			// method 1: doesn't work
			// List<CtElement> children = element.getDirectChildren();
			// CtElement lastChild = children.get(children.size()-1);
			// lastChild.addComment(element.getFactory().Code().createComment("End of for",CtComment.CommentType.INLINE));
			// element.getDirectChildren().set(children.size()-1,lastChild);
			
			// method 2:
			int count = 0;
			CtIterator iterator = new CtIterator(element);
			while (iterator.hasNext()) {
				CtElement el = iterator.next();
				count++;
				System.out.println("Element: "+el);
			}
			int count2=0;
			CtIterator iterator2 = new CtIterator(element);
			while (iterator2.hasNext()) {
				CtElement el = iterator2.next();
				count2++;
				if(count2==count-2)
					el.addComment(element.getFactory().Code().createComment("End of for",CtComment.CommentType.INLINE));
			}
			
			element.addComment(element.getFactory().Code().createComment("Start of for",CtComment.CommentType.INLINE));
		} 
		// else if(lastElementIsFor==true){
			// element.addComment(element.getFactory().Code().createComment("End of for",CtComment.CommentType.INLINE));
			// lastElementIsFor = false;
		// }
	}
}
