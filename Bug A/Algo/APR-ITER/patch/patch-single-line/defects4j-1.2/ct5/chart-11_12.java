--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//// for (int= 0;seg2=d1[seg1]; seg2=d2[seg2]; seg1=d1[seg1]; seg2=d2[seg2]; seg1=d1[seg1]; seg2=d2[seg2];seg1; seg2 =) {{{ seg2 =
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
