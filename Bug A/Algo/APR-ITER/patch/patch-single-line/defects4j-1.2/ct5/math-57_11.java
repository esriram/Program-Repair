--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// Add one newcentroids.// Add the nearest centeraCluster<T> nearest =// final double r =} //// Add a new centerto the nearest centerdouble d = nearestx;d=.nearestfinal doublefinal double d =Doublefinal double d = nearest(;= nearest. getDistanceTo;nearest
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
