--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//// Set thecenter.final double distanceTofinal double r2final double d;Double nearestdouble d2 =double nearest =getNearestCluster ( resultSet, pdouble d2 =) ; if( resultSet. size (>= dx2.double ) dx2 [int) { return;< dx2int i =
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
