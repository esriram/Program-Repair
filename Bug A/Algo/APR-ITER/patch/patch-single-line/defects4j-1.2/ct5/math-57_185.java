--- 
+++ 
@@ -12,7 +12,7 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//// Add one new data= pointSet.remove(random.nextInt(pointSet.size()));doublefinal int i0; i < dx2.length;< dx2[i] >{(i++) {// Find nearest{if (dx2[i]2){ //i++ {DoubleDoubleHashMap=]
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
