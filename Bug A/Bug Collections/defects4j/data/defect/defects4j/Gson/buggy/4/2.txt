  private JsonWriter open(int empty, String openBracket) throws IOException {
    beforeValue(true);
    push(empty);
    out.write(openBracket);
    return this;
  }