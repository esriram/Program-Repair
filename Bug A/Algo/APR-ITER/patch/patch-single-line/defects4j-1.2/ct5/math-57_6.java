--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//k/2)// Thethe center.int i = 0;doublefor (int i< dx2.length; i++) {0; i < dx2[i]{}.doubleint i =) ;final double d =;double d =dfinal double d =double d =d ) {double
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
