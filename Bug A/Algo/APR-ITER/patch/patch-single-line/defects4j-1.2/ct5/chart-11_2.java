--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//if (!Arrays.equals(d1,{}{if (!Arrays.equals(d2, d1)) {return false;// //d2.next();//d1.next(); //d1.next();//d2.next(); //d2.next();=//d1.next(); //d1.
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
