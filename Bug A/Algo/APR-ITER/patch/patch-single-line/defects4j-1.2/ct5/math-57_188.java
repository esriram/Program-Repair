--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//of thea single center.final double d =final double di.kk ) {. add ( new) ;Cluster<T> nearestgetNearestCluster (, nearestclusterCluster < TCluster < T > nearest.nearest. getCenterclusterdistance from the closest center() {;final double
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
