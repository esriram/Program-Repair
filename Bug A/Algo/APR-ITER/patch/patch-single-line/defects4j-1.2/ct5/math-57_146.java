--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// Find the nearestnearest center.Cluster<T> nearest// distance from nearest center.double d =0.0;dx2[i] = d;d) {+= d;final double d =0.0;//double)(2) {if+=} else {d =double d+; }
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
