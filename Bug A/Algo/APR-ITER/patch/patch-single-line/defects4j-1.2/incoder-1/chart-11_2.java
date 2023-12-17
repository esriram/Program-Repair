--- 
+++ 
@@ -9,7 +9,10 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+    PathIterator iterator2 = p2.getPathIterator(null);
+    // We don't know which segments are going to be the same in two cases (e.g.
+    // windingRule != 0), so just check that they were equal in terms of the
+    // winding rules.
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
