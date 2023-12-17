    public List getValues(final Option option,
                          final List defaultValues) {
        // First grab the command line values
        List valueList = (List) values.get(option);

        // Secondly try the defaults supplied to the method
        if ((valueList == null) || valueList.isEmpty()) {
            valueList = defaultValues;
        }

        // Thirdly try the option's default values
        if ((valueList == null) || valueList.isEmpty()) {
            valueList = (List) this.defaultValues.get(option);
        }

        // Finally use an empty list
        if (valueList == null) {
            valueList = Collections.EMPTY_LIST;
        }

        return valueList;
    }