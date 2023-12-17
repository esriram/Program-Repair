--- 
+++ 
@@ -1,6 +1,6 @@
 public MultiplePiePlot(CategoryDataset dataset) {
     super();
-    this.dataset = dataset;
+    setDataset(dataset);
     PiePlot piePlot = new PiePlot(null);
     this.pieChart = new JFreeChart(piePlot);
     this.pieChart.removeLegend();
