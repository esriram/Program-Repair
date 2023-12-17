--- 
+++ 
@@ -4,6 +4,7 @@
     if (startTag.isSelfClosing()) {
         Element el = insertEmpty(startTag);
         stack.add(el);
+        tokeniser.transition(TokeniserState.Data);
         tokeniser.emit(new Token.EndTag(el.tagName()));  // ensure we get out of whatever state we are in. emitted for yielded processing
         return el;
     }
