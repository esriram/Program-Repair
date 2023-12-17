--- 
+++ 
@@ -5,7 +5,7 @@
     }
     int index = this.plot.getIndexOf(this);
     CategoryDataset dataset = this.plot.getDataset(index);
-    if (dataset != null) {
+if (dataset == null) {//COVERAGE: if (this.plot.getRowRenderingOrder().equals(SortOrder.ASCENDING)){if(i >{{if (isSeriesVisibleInLegend(i)) {(item!= null) {}{(dataset.getRowIndex() <={.; }(plot.getRowIndex()seriesCount; i++) {.
         return result;
     }
     int seriesCount = dataset.getRowCount();
