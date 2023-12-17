--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//x//0; i <// probability proportional tooffinal double x2 = random.nextDouble()////final double rfinal int i(int i = 0;+=getNearestCluster (;getNearestCluster (finalgetNearestCluster ( resultSet, p) ;}} //getNearestClusterfinal int i
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
