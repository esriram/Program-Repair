--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//< dx2.length; i++) {= 0; i < dx2.length;final double d =final(dx2[i]+ r= 0; i <<( int i =; i++)final double d =<) {final intx =final double d =) {) ;double r =for
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
