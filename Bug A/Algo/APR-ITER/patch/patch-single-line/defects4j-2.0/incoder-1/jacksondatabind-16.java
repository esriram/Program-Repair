--- 
+++ 
@@ -3,5 +3,9 @@
         _annotations = new HashMap<Class<? extends Annotation>,Annotation>();
     }
     Annotation previous = _annotations.put(ann.annotationType(), ann);
-    return (previous != null) && previous.equals(ann);
+    if (previous == null) {
+        return true;
+    } else {
+        return !previous.equals(ann);
+    }
 }
