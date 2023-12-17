--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;// Add one newnew// Set up theclusteres.Cluster < T > nearestgetNearestCluster ( resultSet, nearestgetNearestCluster ( resultSet, nearest// Compute the nearest centernearest. getCentergetNearestCluster1// Do// Remove one newcluster. A distance//thethe nearestx{Cluster < Ta center.in nearestclosest
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
