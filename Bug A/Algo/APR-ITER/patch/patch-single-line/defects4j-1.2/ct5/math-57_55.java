--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0;// Get randomeach point x.double r2 = random.nextDouble();for (int i0; i <int i< dx2.length; i++) {ifif= 0; i <r2) {2) {}// Nodouble= 0 ;(doubledouble d =)if (= 0 ; i <
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
