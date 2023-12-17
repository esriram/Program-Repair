--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//=getSegmentCount(d1);(d2) {//seg2, seg1, seg2)) {= seg1 == seg2)return false;if (seg1seg2) {return false;(d1){(d2)return} if; return;seg
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
