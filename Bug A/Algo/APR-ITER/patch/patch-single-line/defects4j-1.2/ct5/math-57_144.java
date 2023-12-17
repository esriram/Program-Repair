--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;// Getk) {the points in the// Thetheofthe cluster is////finalT p =get (Cluster<T> nearest( nearestdouble d =d * d;< dx2.length; i++) {(for(i = 0; i <) {final doublefinalfinal double d =1.
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
