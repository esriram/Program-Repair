--- 
+++ 
@@ -5,7 +5,8 @@
     return (cal1.get(Calendar.MILLISECOND) == cal2.get(Calendar.MILLISECOND) &&
             cal1.get(Calendar.SECOND) == cal2.get(Calendar.SECOND) &&
             cal1.get(Calendar.MINUTE) == cal2.get(Calendar.MINUTE) &&
-            cal1.get(Calendar.HOUR) == cal2.get(Calendar.HOUR) &&
+            cal1.get(Calendar.HOUR_OF_DAY) == cal2.get(Calendar.HOUR_OF_DAY) &&
+            cal1.get(Calendar.DAY_OF_WEEK) == cal2.get(Calendar.DAY_OF_WEEK) &&
             cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR) &&
             cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
             cal1.get(Calendar.ERA) == cal2.get(Calendar.ERA) &&
