    public DocumentType(String name, String publicId, String systemId, String baseUri) {
        super(baseUri);

        attr(NAME, name);
        attr(PUBLIC_ID, publicId);
        attr(SYSTEM_ID, systemId);
    }