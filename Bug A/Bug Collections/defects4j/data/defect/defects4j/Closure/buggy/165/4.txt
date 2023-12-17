  RecordType(JSTypeRegistry registry, Map<String, RecordProperty> properties) {
    super(registry, null, null);
    setPrettyPrint(true);

    for (String property : properties.keySet()) {
      RecordProperty prop = properties.get(property);
      if (prop == null) {
        throw new IllegalStateException(
            "RecordProperty associated with a property should not be null!");
      }
        defineDeclaredProperty(
            property, prop.getType(), prop.getPropertyNode());
    }

    // Freeze the record type.
    isFrozen = true;
  }