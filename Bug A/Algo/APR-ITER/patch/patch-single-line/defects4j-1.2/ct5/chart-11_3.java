--- 
+++ 
@@ -9,7 +9,7 @@
         return false;
     }
     PathIterator iterator1 = p1.getPathIterator(null);
-    PathIterator iterator2 = p1.getPathIterator(null);
+PathIterator iterator2 = p2.getPathIterator(null);//System.out.println(iterator1.toString());={if (seg1!= seg2){ return false;if (!Arrays.equals(d1,} else {//System.out.println(seg2); } //System.out.println(d2);= null; };//System.out.println(
     double[] d1 = new double[6];
     double[] d2 = new double[6];
     boolean done = iterator1.isDone() && iterator2.isDone();
