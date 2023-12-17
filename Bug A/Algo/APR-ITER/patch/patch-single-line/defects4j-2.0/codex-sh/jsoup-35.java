--- 
+++ 
@@ -477,6 +477,7 @@
                     }
 
                     Element adopter = new Element(formatEl.tag(), tb.getBaseUri());
+                    adopter.attributes().addAll(formatEl.attributes());
                     Node[] childNodes = furthestBlock.childNodes().toArray(new Node[furthestBlock.childNodeSize()]);
                     for (Node childNode : childNodes) {
                         adopter.appendChild(childNode); // append will reparent. thus the clone to avoid concurrent mod.
