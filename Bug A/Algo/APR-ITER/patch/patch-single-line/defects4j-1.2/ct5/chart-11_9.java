--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//if//if//if (!Arrays.equals(d1, d2)){}//if (!Arrays.equals(d1, d2)){ //if (seg1 ==return} if (seg1!= seg2){ //if (!Arrays.equals(d1, d2))return false; //if//if (
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
