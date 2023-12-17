--- 
+++ 
@@ -6,7 +6,7 @@
     }
     Class<?>[] classes = new Class[array.length];
     for (int i = 0; i < array.length; i++) {
-        classes[i] = array[i].getClass();
+        classes[i] = array[i]!= null? array[i].getClass() : null;
     }
     return classes;
 }
