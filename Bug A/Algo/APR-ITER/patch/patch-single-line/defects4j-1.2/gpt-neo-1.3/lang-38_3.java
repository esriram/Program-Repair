--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        long t = calendar.getTimeInMillis();
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
