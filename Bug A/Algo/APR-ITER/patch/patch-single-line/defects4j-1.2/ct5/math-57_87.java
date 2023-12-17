--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;////points, a centerone data point. //Cluster<T> nearest =int i =< dx2.length; i++) {double d =i ] =. get (.) ;Double.longestOne(dx2[i]){ //newfork=0; i <//the nearestnearest cluster. //+
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
