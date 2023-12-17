--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//if (iterator1.getWindingRule() ==//if (p1.getWindingRule() ==newd2.length) {{returnfalse;= iterator2.getWindingRule();}; if (iterator1.isDone()iterator2.isDone()) {;{if
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
