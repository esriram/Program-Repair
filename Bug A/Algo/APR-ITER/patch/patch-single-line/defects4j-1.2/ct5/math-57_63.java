--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;//final double d= random.nextDouble() *double sum = 0;{ double nearest=cluster.distance between x, nearest.getCenter()// Add one newc//of//getNearestCluster(resultSet, p);final double x =(dx2[i] >= r)d(//2=2 * dx2
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
