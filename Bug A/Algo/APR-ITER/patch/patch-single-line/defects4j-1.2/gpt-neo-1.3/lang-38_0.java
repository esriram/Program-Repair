--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        Date time = calendar.getTime();
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
