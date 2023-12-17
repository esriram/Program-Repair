--- 
+++ 
@@ -6,7 +6,7 @@
     if (x < arr[mid]) {
         return binsearch(arr, x, start, mid);
     } else if (x > arr[mid]) {
-        return binsearch(arr, x, mid, end);
+        return binsearch(arr, x, mid + 1, end);
     } else {
         return mid;
     }
