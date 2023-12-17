    public static <E> Iterator<E> collatedIterator(final Comparator<? super E> comparator,
                                                   final Collection<Iterator<? extends E>> iterators) {
        return new CollatingIterator<E>(comparator, iterators);
    }