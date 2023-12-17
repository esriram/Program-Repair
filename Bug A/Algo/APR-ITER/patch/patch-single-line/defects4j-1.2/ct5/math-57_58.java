--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//int i0; i < dx2.length; i++)forint j< dx2.length; i++) {final Cluster<T> nearestif (dx2[i] <{Cluster<T> nearest = getNearestCluster(resultSet,//=) {//nearest) {=xd =) *)
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
