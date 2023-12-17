--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;////the nearest center.double. size< dx2. length ;double d = 0.0}i = 0 ; iresultSet.add(newfinal double r =; if (dx2[i]r) {double] =) {final double d == d /{ final T p =dx2 [ ifinal double
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
