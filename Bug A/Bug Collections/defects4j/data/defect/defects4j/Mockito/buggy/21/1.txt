    private static <T> InstantationException paramsException(Class<T> cls, Exception e) {
        return new InstantationException("Unable to create mock instance of '"
                + cls.getSimpleName() + "'.\nPlease ensure that the outer instance has correct type and that the target class has parameter-less constructor.", e);
    }