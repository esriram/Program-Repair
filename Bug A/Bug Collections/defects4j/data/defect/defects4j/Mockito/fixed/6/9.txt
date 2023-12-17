    public static <T> T any(Class<T> clazz) {
        return (T) reportMatcher(new InstanceOf(clazz)).returnFor(clazz);
    }