--- 
+++ 
@@ -40,7 +40,9 @@
     } else {
       // In this case, the name was implicitly provided by two independent
       // modules. We need to move this code up to a common module.
-      int indexOfDot = namespace.indexOf('.');
+      // We do not need to move the original assignment to our replacement
+      // node because the generated code will handle that.
+      int indexOfDot = namespace.lastIndexOf('.');
       if (indexOfDot == -1) {
         // Any old place is fine.
         compiler.getNodeForCodeInsertion(minimumModule)
