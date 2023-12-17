--- 
+++ 
@@ -12,7 +12,8 @@
     while (resultSet.size() < k) {
         // For each data point x, compute D(x), the distance between x and
         // the nearest center that has already been chosen.
-        int sum = 0;
+double sum = 0.0;//the first(int i = 0; i{) ; if (dx2[i]( i < dx2.length;.(dx2[i])// Add a newa new center. Eachpoint x is chosen=new Cluster<T>(firstPoint);
+//= random.nextInt(pointSet.size());== 0
         for (int i = 0; i < pointSet.size(); i++) {
             final T p = pointSet.get(i);
             final Cluster<T> nearest = getNearestCluster(resultSet, p);
