--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//// Use the nearestcluster if necessary.)//= 0; i < dx2.length;// Add a newcenter withfinal double= 0.0;;ifdx2[i] >= r) {a new data pointthe// Set//final double= 0.0; i={( double= 0
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
