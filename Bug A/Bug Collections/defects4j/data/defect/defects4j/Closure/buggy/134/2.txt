    public void addNode(Property prop) {
      typesInSet.or(prop.typesSet);
      typesRelatedToSet.or(getRelated(prop.type));
    }