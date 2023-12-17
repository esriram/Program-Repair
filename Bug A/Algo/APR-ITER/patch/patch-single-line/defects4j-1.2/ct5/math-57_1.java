--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// If no nearest centerreturncluster. // IfpointSet.size() > 0)double score =i = 0;0; i <) {Cluster<T> nearest =.{i} ////1double score =(; i ++{ if (dx2[i] >=score >= 2.0score = Math
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
