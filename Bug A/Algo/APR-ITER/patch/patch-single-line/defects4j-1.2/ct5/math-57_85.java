--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//inpoints andthe nearest center.adistance.the nearestthe// the nearest center isfinal double sum =double d =. size (; igetNearestCluster () {final; final double=( double)= pointSet.get(i);;if (dx2[i] >= r)resultSet
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
