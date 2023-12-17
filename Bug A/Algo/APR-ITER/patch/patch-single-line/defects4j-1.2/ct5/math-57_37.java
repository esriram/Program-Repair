--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// Find the nearest center.final doublefinal double max =doublemax; // double=//for (int i =) {.;} //finalfinal doublemaxfinal double//) {= resultSet.get(i);cluster.add(new Cluster<T>(nearest.getCenter())); // double//double )
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
