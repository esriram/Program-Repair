--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int hours = 24 - calendar.get(Calendar.HOUR_OF_DAY);
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
