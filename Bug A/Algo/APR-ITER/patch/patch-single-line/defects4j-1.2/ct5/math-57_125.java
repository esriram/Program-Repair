--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;// No1newfor (int i< dx2.length; i++) {(dx2[i] -2 ) {(++ {)( i = 0(i++) {Double>=i <) {Cluster<T>); final doubledouble d =Math. sqrt ( d= Math. max;
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
