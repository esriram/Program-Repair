--- 
+++ 
@@ -12,7 +12,9 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+        // Choose any point in the data set whose closest point to x has not
+        // already been chosen.
+        double sum = 0;
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
