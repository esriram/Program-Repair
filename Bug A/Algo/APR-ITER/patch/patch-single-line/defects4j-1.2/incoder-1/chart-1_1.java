--- 
+++ 
@@ -5,7 +5,10 @@
     }
     int index = this.plot.getIndexOf(this);
     CategoryDataset dataset = this.plot.getDataset(index);
-    if (dataset != null) {
+    if (dataset == null) {
+        return result;
+    }
+    if (dataset.getRowCount() == 0) {
         return result;
     }
     int seriesCount = dataset.getRowCount();
