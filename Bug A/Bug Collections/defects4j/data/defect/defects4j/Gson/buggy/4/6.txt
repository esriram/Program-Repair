  public JsonWriter value(boolean value) throws IOException {
    writeDeferredName();
    beforeValue(false);
    out.write(value ? "true" : "false");
    return this;
  }