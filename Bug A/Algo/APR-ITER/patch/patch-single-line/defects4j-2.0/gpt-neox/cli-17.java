--- 
+++ 
@@ -19,6 +19,7 @@
         else if (stopAtNonOption)
         {
             process(token.substring(i));
+            return;
         }
         else
         {
