        public void serializeWithType(Object value, JsonGenerator gen,
                SerializerProvider provider, TypeSerializer typeSer) throws IOException {
            // no type info, just regular serialization
            serialize(value, gen, provider);            
        }