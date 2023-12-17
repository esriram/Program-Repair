--- 
+++ 
@@ -38,6 +38,6 @@
     }
     else
     {
-        return null;
+        throw new ParseException("Unable to create value of type " + clazz.getName());
     }
 }
