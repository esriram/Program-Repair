--- 
+++ 
@@ -9,7 +9,8 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//if (!Arrays.equals(d1,{{
+//} // if// if (seg1 == seg2)// if (seg1 ==seg2!= seg1) {return false;(d1, d2);// if (seg2 == seg1){} if (seg2 == seg2)
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
