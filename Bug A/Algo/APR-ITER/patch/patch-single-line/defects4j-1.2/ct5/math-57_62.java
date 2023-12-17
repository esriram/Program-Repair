--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//= random.nextDouble();=Cluster<T> nearest =( nearest.getCenter());Cluster<T> nearest ==) ) ;(; final doubledx2 [ i+= r//if (dx2[i] >= r) {= 1.0; //double d =// //// if//to be chosen
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
