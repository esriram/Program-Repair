--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        StringBuffer tmp = new StringBuffer(calendar.get(Calendar.YEAR));
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
