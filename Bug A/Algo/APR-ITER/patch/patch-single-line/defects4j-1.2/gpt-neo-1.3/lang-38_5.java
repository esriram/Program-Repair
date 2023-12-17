--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int offset = calendar.get(Calendar.DAY_OF_MONTH) + 1;
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
