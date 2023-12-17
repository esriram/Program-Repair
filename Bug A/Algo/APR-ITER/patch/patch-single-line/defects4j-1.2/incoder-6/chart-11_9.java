--- 
+++ 
@@ -9,7 +9,12 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+    PathIterator iterator2 = p2.getPathIterator(null);
+    int comp1 = p1.getWindingRule();
+    int comp2 = p2.getWindingRule();
+    if (comp1 != comp2) {
+        return false;
+    }
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
