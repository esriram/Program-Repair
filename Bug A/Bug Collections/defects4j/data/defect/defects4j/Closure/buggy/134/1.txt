    public boolean isIndependentOf(Property prop) {
      if (typesRelatedToSet.intersects(prop.typesSet)) {
        return false;
      }
      return !getRelated(prop.type).intersects(typesInSet);
    }