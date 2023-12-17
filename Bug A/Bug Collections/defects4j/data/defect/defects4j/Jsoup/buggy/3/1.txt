    public Element prepend(String html) {
        Validate.notNull(html);
        
        Element fragment = Parser.parseBodyFragment(html, baseUri).body();
        List<Node> nodes = fragment.childNodes();
        for (int i = nodes.size() - 1; i >= 0; i--) {
            Node node = nodes.get(i);
            node.parentNode = null;
            prependChild(node);
        }
        return this;
    }