--- 
+++ 
@@ -38,6 +38,6 @@
     }
     else
     {
-        return null;
+        throw new ParseException("You must use a different pattern option for the same file in your project.");
     }
 }
