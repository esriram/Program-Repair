--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int tzHour = calendar.get(Calendar.HOUR_OF_DAY);
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
