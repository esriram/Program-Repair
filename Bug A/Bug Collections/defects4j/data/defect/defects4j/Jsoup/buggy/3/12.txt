    private Tag setAncestor(String... tagNames) {
        if (tagNames == null) {
            ancestors = Collections.emptyList();
        } else {
            ancestors = new ArrayList<Tag>(tagNames.length);
            for (String name : tagNames) {
                ancestors.add(Tag.valueOf(name));
            }
        }
        return this;
    }