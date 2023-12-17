  public boolean hasProperty(String name) {
    return super.hasProperty(name) || "prototype".equals(name);
  }