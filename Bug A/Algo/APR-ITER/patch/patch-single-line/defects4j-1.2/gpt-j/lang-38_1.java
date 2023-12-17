--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int off = calendar.get(Calendar.ZONE_OFFSET);
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
