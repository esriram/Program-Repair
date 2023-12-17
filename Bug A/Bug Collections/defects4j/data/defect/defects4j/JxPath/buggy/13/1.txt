    public synchronized String getNamespaceURI(String prefix) {

    /**
     * Given a prefix, returns an externally registered namespace URI.
     * 
     * @param prefix The namespace prefix to look up
     * @return namespace URI or null if the prefix is undefined.
     * @since JXPath 1.3
     */
        String uri = (String) namespaceMap.get(prefix);
        if (uri == null && pointer != null) {
            uri = pointer.getNamespaceURI(prefix);
        }
        if (uri == null && parent != null) {
            return parent.getNamespaceURI(prefix);
        }
        return uri;
    }