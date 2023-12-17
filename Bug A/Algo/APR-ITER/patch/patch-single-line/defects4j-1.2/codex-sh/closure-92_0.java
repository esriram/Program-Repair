--- 
+++ 
@@ -40,7 +40,9 @@
     } else {
       // In this case, the name was implicitly provided by two independent
       // modules. We need to move this code up to a common module.
-      int indexOfDot = namespace.indexOf('.');
+      // We'll just put it after the parent namespace.
+      // This can be improved if necessary, but it's not common enough to matter.
+      int indexOfDot = namespace.lastIndexOf('.');
       if (indexOfDot == -1) {
         // Any old place is fine.
         compiler.getNodeForCodeInsertion(minimumModule)
