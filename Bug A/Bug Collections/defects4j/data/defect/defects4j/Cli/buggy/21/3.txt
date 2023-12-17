    public WriteableCommandLineImpl(final Option rootOption,
                                    final List arguments) {
        this.prefixes = rootOption.getPrefixes();
        this.normalised = arguments;
    }