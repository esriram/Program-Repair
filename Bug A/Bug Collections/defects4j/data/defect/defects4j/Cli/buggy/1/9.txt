    public Option[] getOptions()
    {
        Collection processed = options.values();

        // reinitialise array
        Option[] optionsArray = new Option[processed.size()];

        // return the array
        return (Option[]) processed.toArray(optionsArray);
    }