--- 
+++ 
@@ -12,7 +12,10 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+        // For each new data point x, the probability that it is chosen is
+        // proportional to its squared distance.
+        double sum = 0;
+        // For each data point x, D(x) = distance(x)^2.
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
