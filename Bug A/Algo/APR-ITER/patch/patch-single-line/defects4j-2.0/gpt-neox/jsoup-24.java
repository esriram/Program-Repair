--- 
+++ 
@@ -3,7 +3,7 @@
         String name = r.consumeLetterSequence();
         t.tagPending.appendTagName(name.toLowerCase());
         t.dataBuffer.append(name);
-        r.advance();
+        // TODO: Should be able to use the same tag name as the next one
         return;
     }
 
