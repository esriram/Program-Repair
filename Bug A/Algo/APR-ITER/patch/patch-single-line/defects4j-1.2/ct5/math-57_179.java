--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//a random point.x, compute D(x), the distancepoint//to do thisif) {{ double=if= 0; i < dx2.length;<++ {new; }= 0 ; i <) {(} }}doublefinal int i = 0) {= dx2 [
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
