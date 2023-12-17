  public JsonWriter value(String value) throws IOException {
    if (value == null) {
      return nullValue();
    }
    writeDeferredName();
    beforeValue(false);
    string(value);
    return this;
  }