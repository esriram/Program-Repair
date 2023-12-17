--- 
+++ 
@@ -9,7 +9,13 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+    PathIterator iterator2 = p2.getPathIterator(null);
+    if (iterator1.isDone() != iterator2.isDone()) {
+        return false;
+    }
+    if (iterator1.isDone()) {
+        return true;
+    }
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
