  public JsonWriter value(long value) throws IOException {
    writeDeferredName();
    beforeValue(false);
    out.write(Long.toString(value));
    return this;
  }