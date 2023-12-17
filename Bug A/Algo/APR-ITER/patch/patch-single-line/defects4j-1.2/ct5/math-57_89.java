--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// Acluster.are) {; //) ; final( double(0 ; i <dx2[i]}++ ) {final int d =)d <d ) {=( ) ;)= p.distanceFrom(nearest.getCenter());if (dx2[i] < d){final double
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
