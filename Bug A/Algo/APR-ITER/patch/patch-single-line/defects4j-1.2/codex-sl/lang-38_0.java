--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int gmtOffset = mTimeZone.getOffset(calendar.getTimeInMillis());
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
