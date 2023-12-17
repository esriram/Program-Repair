--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;//inrandom numberto choose thethe initialfinal; //theandthe // nearest center.final) { final T;final double( inti =< dx2.length; i++) {=) {;; } //k; } //{of a point. //
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
