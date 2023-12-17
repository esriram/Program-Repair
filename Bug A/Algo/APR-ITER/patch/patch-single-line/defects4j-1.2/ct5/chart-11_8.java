--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//in the sameifiterator2.isDone()){ if (seg1!= seg2)returnreturntrue;}== p2.getPathIterator(null); //!=((seg1!= seg2)if (!Arrays.equals(d1, d2)) {return false;== p2.get
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
