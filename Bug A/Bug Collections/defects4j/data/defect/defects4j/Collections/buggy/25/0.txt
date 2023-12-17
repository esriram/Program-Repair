    public static <E> Iterator<E> collatedIterator(final Comparator<? super E> comparator,
                                                   final Iterator<? extends E> iterator1,
                                                   final Iterator<? extends E> iterator2) {
        return new CollatingIterator<E>(comparator, iterator1, iterator2);
    }