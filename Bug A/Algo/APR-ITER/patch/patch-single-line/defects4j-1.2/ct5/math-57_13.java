--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;// Remove one//double d ==getNearestCluster( resultSet, firstPoint=) ; // Ifthedistanceina center. //1if (dx2[i] >=++ ) { //final int i = 0;=int i = 0= 0; i <{ ////if (dx2[i] <===
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
